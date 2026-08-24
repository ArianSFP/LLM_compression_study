from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oracle_study.d1_tail_quality import (
    D1_PREFIX,
    FIXED_D1_POLICY,
    HISTORICAL_PR13,
    LOCAL_POLICY,
    OracleCell,
    PR13_POLICY,
    REPAIRED_D1_POLICY,
    RERANKED_D1_POLICY,
    STANDARD_POLICIES,
    add_live_minus_frozen,
    assemble_policy_delta_bank,
    calibrate_fixed_d1_policy,
    downstream_tail_metrics,
    request_delta,
    request_policy_local_metrics,
    sha256,
    validate_expansion_config,
    validate_tail_config,
    verify_sha256_manifest,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_frozen_expansion_and_tail_configs_validate() -> None:
    expansion = json.loads((
        EXPERIMENT / "configs/qwen36_mxfp4_d1_layer_slice_expansion_20260824_v1.json"
    ).read_text())
    tail = json.loads((
        EXPERIMENT / "configs/qwen36_mxfp4_d1_full_sequence_tail_kl_legacy_20260824_v1.json"
    ).read_text())
    validate_expansion_config(expansion)
    validate_tail_config(tail)
    changed_eta = deepcopy(tail)
    changed_eta["fixed_d1_calibration"]["candidate_eta"][-1] = 0.01
    with np.testing.assert_raises_regex(ValueError, "eta grid changed"):
        validate_tail_config(changed_eta)


def test_sha256_manifest_verifies_every_safe_relative_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"immutable")
    manifest = tmp_path / "artifact_hashes.sha256"
    manifest.write_text(f"{sha256(artifact)}  {artifact.name}\n")
    assert verify_sha256_manifest(tmp_path, manifest) == 1
    artifact.write_bytes(b"changed")
    with np.testing.assert_raises_regex(RuntimeError, "artifact changed"):
        verify_sha256_manifest(tmp_path, manifest)


def _synthetic_cell(layer: int, rate: int, preferred: str = "d1_strict_eta_b") -> OracleCell:
    policies = ["d1_strict_eta_a", "d1_strict_eta_b"]
    metric_rows = []
    allocation_rows = []
    for policy in policies:
        for group, request_id in enumerate(("request-a", "request-b")):
            crossing = policy != preferred
            metric_rows.append({
                "layer": layer,
                "rate_pages_per_expert": rate,
                "group": group,
                "request_id": request_id,
                "position": 0,
                "policy": policy,
                "exact_d1_crossed": crossing,
                "membership_pairs_changed": int(crossing),
                "routing_mass_lost": 0.1 if crossing else 0.0,
                "local_qenergy_damage": 2.0 if policy == preferred else 1.0,
            })
            allocation_rows.append({
                "layer": layer,
                "rate_pages_per_expert": rate,
                "group": group,
                "request_id": request_id,
                "position": 0,
                "policy": policy,
                "selected_group_pages": rate * 8,
            })
    for policy, damage in (
        (HISTORICAL_PR13, 1.5),
        ("exact_combined_local_column_generated_frontier", 1.25),
    ):
        for group, request_id in enumerate(("request-a", "request-b")):
            metric_rows.append({
                "layer": layer,
                "rate_pages_per_expert": rate,
                "group": group,
                "request_id": request_id,
                "position": 0,
                "policy": policy,
                "exact_d1_crossed": False,
                "membership_pairs_changed": 0,
                "routing_mass_lost": 0.0,
                "local_qenergy_damage": damage,
            })
    allocation_rows.extend([
        {
            "layer": layer,
            "rate_pages_per_expert": rate,
            "group": group,
            "request_id": request_id,
            "position": 0,
            "policy": "exact_combined_local_column_generated_frontier",
            "selected_group_pages": rate * 8,
        }
        for group, request_id in enumerate(("request-a", "request-b"))
    ])
    zeros = np.zeros((2, 2048), np.float32)
    deltas = {
        f"{HISTORICAL_PR13}__rate_{rate}": zeros + 1.0,
        f"exact_combined_local_column_generated_frontier__rate_{rate}": zeros + 2.0,
        f"d1_strict_eta_a__rate_{rate}": zeros + 3.0,
        f"d1_strict_eta_b__rate_{rate}": zeros + 4.0,
    }
    return OracleCell(
        layer=layer,
        rate=rate,
        directory=Path("/synthetic"),
        facts={"groups": 2},
        metrics=pd.DataFrame(metric_rows),
        allocations=pd.DataFrame(allocation_rows),
        deltas=deltas,
    )


def test_calibration_prioritizes_crossings_before_local_damage() -> None:
    cells = {
        (0, 384): _synthetic_cell(0, 384),
        (12, 384): _synthetic_cell(12, 384),
    }
    selected, evidence = calibrate_fixed_d1_policy(cells, [0, 12], [384])
    assert selected == {384: "d1_strict_eta_b"}
    assert evidence[evidence["selected"]]["exact_crossings"].item() == 0


def test_policy_bank_assembles_fixed_reranked_and_repair(tmp_path: Path) -> None:
    cell = _synthetic_cell(1, 384)
    repair = tmp_path / "repair"
    repair.mkdir()
    repaired = np.full((2, 2048), 5.0, np.float32)
    delta_path = repair / "d1_exact_repair_delta.npz"
    np.savez_compressed(delta_path, delta=repaired)
    repair_metrics = repair / "d1_exact_repair_metrics.parquet"
    pd.DataFrame({
        "group": [0, 1],
        "local_qenergy_damage": [2.5, 2.5],
    }).to_parquet(repair_metrics, index=False)
    (repair / "d1_exact_repair_facts.json").write_text(json.dumps({
        "layer": 1,
        "rate_pages_per_expert": 384,
        "outputs": {
            delta_path.name: {"sha256": sha256(delta_path)},
            repair_metrics.name: {"sha256": sha256(repair_metrics)},
        },
    }))
    bank = assemble_policy_delta_bank(cell, repair, "d1_strict_eta_b")
    assert tuple(bank.deltas) == STANDARD_POLICIES
    np.testing.assert_array_equal(bank.deltas[PR13_POLICY], 1.0)
    np.testing.assert_array_equal(bank.deltas[LOCAL_POLICY], 2.0)
    np.testing.assert_array_equal(bank.deltas[FIXED_D1_POLICY], 4.0)
    np.testing.assert_array_equal(bank.deltas[RERANKED_D1_POLICY], 4.0)
    np.testing.assert_array_equal(bank.deltas[REPAIRED_D1_POLICY], 5.0)
    request, positions = request_delta(bank, PR13_POLICY, "request-b", 3)
    np.testing.assert_array_equal(positions, [0])
    np.testing.assert_array_equal(request[0], 1.0)
    np.testing.assert_array_equal(request[1:], 0.0)
    local = request_policy_local_metrics(bank, REPAIRED_D1_POLICY, "request-b")
    assert local["selected_local_qenergy_damage_sum"] == 2.5
    assert local["selected_local_qenergy_damage_per_group"] == 2.5
    assert local["injected_delta_mse"] == 25.0


def test_downstream_metrics_detect_d1_crossing_and_terminal_error() -> None:
    sequence = 3
    experts = 10
    reference_router = {}
    candidate_router = {}
    for layer in range(1, 40):
        base = np.tile(np.arange(experts, 0, -1, dtype=np.float32), (sequence, 1))
        candidate = base.copy()
        if layer == 1:
            candidate[0, 8] = 20.0
        reference_router[layer] = base
        candidate_router[layer] = candidate
    reference_logits = np.zeros((sequence, 7), np.float32)
    candidate_logits = reference_logits.copy()
    candidate_logits[0, 1] = 0.25
    metrics = downstream_tail_metrics(
        reference_logits=reference_logits,
        candidate_logits=candidate_logits,
        input_ids=np.asarray([0, 1, 2]),
        reference_final_hidden=np.zeros((sequence, 4), np.float32),
        candidate_final_hidden=np.ones((sequence, 4), np.float32),
        reference_router=reference_router,
        candidate_router=candidate_router,
        injection_layer=0,
        admitted_positions=[0, 1],
    )
    assert metrics["logit_kl"] > 0.0
    assert metrics["final_hidden_mse"] == 1.0
    assert metrics["d1_route_membership_change_fraction"] == 0.5
    assert metrics["first_route_membership_change_layer"] == 1


def test_live_minus_frozen_is_paired_by_complete_identity() -> None:
    rows = []
    for policy, live, frozen in (("a", 3.0, 1.0), ("b", 2.0, 2.5)):
        for route_mode, value in (("live", live), ("fully_frozen", frozen)):
            rows.append({
                "injection_layer": 4,
                "request_id": "request",
                "rate_pages_per_expert": 384,
                "policy": policy,
                "route_mode": route_mode,
                "logit_kl": value,
            })
    result = add_live_minus_frozen(pd.DataFrame(rows))
    paired = result.groupby("policy")["live_minus_fully_frozen_logit_kl"].first()
    assert paired["a"] == 2.0
    assert paired["b"] == -0.5
