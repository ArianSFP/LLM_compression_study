from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_d1_nested_outcomes as analyzer  # noqa: E402
from oracle_study.d1_nested_artifacts import (  # noqa: E402
    ALLOCATION_MANIFEST_SCHEMA,
    canonical_sha256,
    encode_state,
    freeze_calibration_spec,
    write_sealed_allocation_manifest,
)


def _state(*bits: tuple[int, int, int]) -> list[list[int]]:
    result = [[0] * 512 for _ in range(8)]
    for expert, unit, bit in bits:
        result[expert][unit] |= bit
    return result


def _moves(*bits: tuple[int, int, int]) -> list[dict[str, object]]:
    names = {1: "down", 2: "up", 4: "gate"}
    current: dict[tuple[int, int], int] = {}
    rows = []
    for step, (expert, unit, bit) in enumerate(bits):
        source = current.get((expert, unit), 0)
        destination = source | bit
        rows.append({
            "step": step,
            "expert": expert,
            "expert_id": expert + 10,
            "unit": unit,
            "bit": bit,
            "projection": names[bit],
            "source_state": source,
            "destination_state": destination,
            "page_delta": 1,
        })
        current[(expert, unit)] = destination
    return rows


def _reference(row: dict[str, object]) -> dict[str, object]:
    return {key: row[key] for key in ("arm", "rate", "layer", "request_id")}


def _minimal_manifest() -> dict[str, object]:
    frozen = freeze_calibration_spec({"window": 16, "eta": 0.001})
    allocations = []
    for rate, bits in (
        (360, ((0, 0, 1),)),
        (384, ((0, 0, 1), (1, 1, 2))),
    ):
        cap = rate * 8
        allocations.append({
            "arm": analyzer.ARM_D1,
            "rate": rate,
            "layer": 0,
            "request_id": "r0",
            "split": "evaluation",
            "prompt_sha256": "1" * 64,
            "domain": "code",
            "position": 3,
            "calibration_spec_sha256": frozen["sha256"],
            "expert_ids": list(range(10, 18)),
            "freeze_state": encode_state(_state(), page_cap=cap),
            "selected_state": encode_state(_state(*bits), page_cap=cap),
            "moves": _moves(*bits),
        })
    return {
        "schema": ALLOCATION_MANIFEST_SCHEMA,
        "frozen_calibration": frozen,
        "allocations": allocations,
        "nesting_chains": [{
            "name": "nested",
            "members": [_reference(row) for row in allocations],
        }],
    }


def test_analysis_reauthenticates_seal_and_outcome_identity_pin(tmp_path: Path) -> None:
    manifest_path = tmp_path / "allocations.json"
    seal_path = tmp_path / "allocations.seal.json"
    seal = write_sealed_allocation_manifest(
        _minimal_manifest(), manifest_path=manifest_path, seal_path=seal_path,
    )
    manifest = analyzer.load_json(manifest_path)
    identities = canonical_sha256([
        [
            row["arm"], row["rate"], row["layer"], row["request_id"],
            row["selected_state"]["sha256"],
        ]
        for row in sorted(
            manifest["allocations"],
            key=lambda item: (
                item["arm"], item["rate"], item["layer"], item["request_id"],
            ),
        )
    ])
    facts = {"input_pins": {
        "allocation_manifest_sha256": seal["allocation_manifest_sha256"],
        "frozen_calibration_spec_sha256": seal["frozen_calibration_spec_sha256"],
        "allocation_identities_sha256": identities,
        "allocations": 2,
    }}
    loaded = analyzer._load_authenticated_manifest(facts, manifest_path, seal_path)
    assert len(loaded["allocations"]) == 2

    changed = deepcopy(facts)
    changed["input_pins"]["allocation_identities_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="identities"):
        analyzer._load_authenticated_manifest(changed, manifest_path, seal_path)

    quality = pd.DataFrame([
        {
            "split": row["split"],
            "prompt_sha256": row["prompt_sha256"],
            "domain": row["domain"],
            "position": row["position"],
            "arm": row["arm"],
            "rate_pages_per_expert": row["rate"],
            "injection_layer": row["layer"],
            "request_id": row["request_id"],
            "selected_state_sha256": row["selected_state"]["sha256"],
            "allocation_manifest_sha256": seal["allocation_manifest_sha256"],
        }
        for row in loaded["allocations"]
    ])
    analyzer._require_quality_allocation_identity(
        quality, loaded, seal["allocation_manifest_sha256"],
    )
    quality.loc[0, "selected_state_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sealed allocation identities"):
        analyzer._require_quality_allocation_identity(
            quality, loaded, seal["allocation_manifest_sha256"],
        )

    for column, replacement in (
        ("split", "calibration"),
        ("prompt_sha256", "2" * 64),
        ("domain", "reasoning"),
        ("position", 4),
    ):
        changed_quality = pd.DataFrame([
            {
                "split": row["split"],
                "prompt_sha256": row["prompt_sha256"],
                "domain": row["domain"],
                "position": row["position"],
                "arm": row["arm"],
                "rate_pages_per_expert": row["rate"],
                "injection_layer": row["layer"],
                "request_id": row["request_id"],
                "selected_state_sha256": row["selected_state"]["sha256"],
                "allocation_manifest_sha256": seal["allocation_manifest_sha256"],
            }
            for row in loaded["allocations"]
        ])
        changed_quality.loc[0, column] = replacement
        with pytest.raises(RuntimeError, match="sealed allocation identities"):
            analyzer._require_quality_allocation_identity(
                changed_quality, loaded, seal["allocation_manifest_sha256"],
            )


def _config(requests: int = 24, layers: tuple[int, ...] = (0, 6)) -> dict[str, object]:
    return {
        "arms": [
            analyzer.ARM_PR13, analyzer.ARM_STRICT, analyzer.ARM_LOCAL,
            analyzer.ARM_D1, analyzer.ARM_SEVERITY,
        ],
        "traffic_pairs": [
            {
                "metadata_matched_pages_per_expert": 360,
                "pr13_reference_pages_per_expert": 384,
            },
            {
                "metadata_matched_pages_per_expert": 725,
                "pr13_reference_pages_per_expert": 749,
            },
        ],
        "injection_layers": list(layers),
        "request_source": {"evaluation_requests": requests},
        "statistics": {
            "cluster_bootstrap_resamples": 10_000,
            "paired_sign_flip_resamples": 100_000,
            "bootstrap_seed": 20260825,
        },
        "promotion_gates": {
            "paired_p95_kl_regression_maximum": 0.0,
            "paired_single_request_kl_regression_maximum": 0.002839,
        },
    }


def _request_means(requests: int = 24) -> pd.DataFrame:
    values = {
        (analyzer.ARM_PR13, 384): 4.0,
        (analyzer.ARM_PR13, 749): 4.0,
        (analyzer.ARM_LOCAL, 360): 3.0,
        (analyzer.ARM_LOCAL, 725): 3.0,
        (analyzer.ARM_D1, 360): 1.0,
        (analyzer.ARM_D1, 725): 1.0,
    }
    rows = []
    for request in range(requests):
        for arm in _config()["arms"]:
            for rate in (360, 384, 725, 749):
                rows.append({
                    "split": "evaluation",
                    "request_id": f"r{request:03d}",
                    "arm": arm,
                    "rate_pages_per_expert": rate,
                    "route_mode": "live",
                    "domain": "code" if request % 2 else "reasoning",
                    "logit_kl": values.get((arm, rate), 2.0),
                })
    return pd.DataFrame(rows)


def test_primary_inference_uses_declared_four_test_family_and_separates_ci_holm() -> None:
    paired, inference = analyzer._primary_request_inference(
        _request_means(), _config(),
    )
    assert len(paired) == 4 * 24
    assert inference["contrast"].tolist() == [spec[0] for spec in analyzer.PRIMARY_SPECS]
    assert inference["bootstrap_resamples"].eq(10_000).all()
    assert inference["sign_flip_permutations"].eq(100_000).all()
    assert inference["sign_flip_alternative"].eq("less").all()
    assert not inference["bootstrap_interval_multiplicity_adjusted"].any()
    assert inference["nominal_ci_upper_below_zero"].all()
    assert inference["holm_sign_flip_p_below_0_05"].all()
    assert np.all(
        inference["sign_flip_p_holm"].to_numpy()
        >= inference["sign_flip_p_raw"].to_numpy()
    )


def test_secondary_contrasts_and_live_frozen_distributions_are_descriptive() -> None:
    paired, summary = analyzer._secondary_request_summaries(
        _request_means(), _config(),
    )
    assert len(paired) == 3 * 4 * 24
    assert len(summary) == 3 * 4
    assert set(summary["contrast_family"]) == {
        family[0] for family in analyzer.SECONDARY_FAMILIES
    }
    assert summary["inferential_status"].eq(
        "secondary_descriptive_no_promotion_use"
    ).all()

    contrast_rows = []
    for request in range(24):
        for arm in _config()["arms"]:
            for rate in (360, 384, 725, 749):
                contrast_rows.append({
                    "split": "evaluation",
                    "request_id": f"r{request:03d}",
                    "arm": arm,
                    "rate_pages_per_expert": rate,
                    "live_minus_fully_frozen_logit_kl": request / 1000.0,
                })
    live_frozen = analyzer._live_frozen_distribution_summary(
        pd.DataFrame(contrast_rows),
    )
    assert len(live_frozen) == 5 * 4
    assert not live_frozen["used_for_promotion"].any()
    assert live_frozen["inferential_status"].eq(
        "paired_descriptive_failure_interpretation"
    ).all()


def test_standalone_v2_config_full_schema_is_frozen() -> None:
    path = ROOT / "configs" / analyzer.MATERIALIZED_CONFIG_NAME
    config = analyzer.load_json(path)
    analyzer.validate_outcome_config(config)
    analyzer._validate_materialized_v2_config(config)

    missing = deepcopy(config)
    del missing["atomic_resume"]
    with pytest.raises(RuntimeError, match="top-level schema"):
        analyzer._validate_materialized_v2_config(missing)

    changed = deepcopy(config)
    changed["request_source"]["evaluation_requests"] = 95
    with pytest.raises(RuntimeError, match="request split"):
        analyzer._validate_materialized_v2_config(changed)


def _quality(requests: int = 2) -> pd.DataFrame:
    rows = []
    for request in range(requests):
        for layer in (0, 6):
            for arm in _config()["arms"]:
                for rate in (360, 384, 725, 749):
                    value = float(layer + request)
                    if arm == analyzer.ARM_D1 and rate in (360, 725):
                        value -= 1.0
                    rows.append({
                        "split": "evaluation",
                        "request_id": f"r{request}",
                        "injection_layer": layer,
                        "position": 3,
                        "domain": "code" if request else "reasoning",
                        "arm": arm,
                        "rate_pages_per_expert": rate,
                        "route_mode": "live",
                        "logit_kl": value,
                    })
    return pd.DataFrame(rows)


def test_layer_domain_and_d1_next_mixer_strata_are_request_paired() -> None:
    layer = analyzer._layer_differences(_quality(), _config(2, (0, 6)))
    assert len(layer) == 4 * 2 * 2
    assert set(layer["d1_next_layer_mixer"]) == {
        "linear_attention", "full_attention",
    }
    strata = analyzer._strata_summary(layer)
    assert set(strata["stratum_type"]) == {"domain", "layer", "attention"}
    attention = strata[strata["stratum_type"].eq("attention")]
    assert set(attention["stratum"]) == {"linear_attention", "full_attention"}
    assert attention["inferential_status"].eq(
        "descriptive_only_no_multiplicity_claim"
    ).all()


def _allocation(
    frozen: str,
    *,
    arm: str,
    rate: int,
    bits: tuple[tuple[int, int, int], ...],
    crossings: int,
    local_damage: float,
    endpoint: str,
) -> dict[str, object]:
    cap = rate * 8
    parameter: dict[str, object] = {"candidate": {"endpoint": endpoint}}
    if arm in {analyzer.ARM_LOCAL, analyzer.ARM_D1, analyzer.ARM_SEVERITY}:
        parameter["frozen"] = {
            "repair_window_pages": 16,
            "eta": 0.001,
        }
    if arm in {analyzer.ARM_D1, analyzer.ARM_SEVERITY}:
        parameter["candidate"] = {
            "endpoint": endpoint,
            "eta": 0.001,
            "pair_fallback_high_guardrail_infeasible": endpoint == "low",
            f"{endpoint}_stop_reason": "no_useful_page",
        }
    return {
        "arm": arm,
        "rate": rate,
        "layer": 0,
        "request_id": "r0",
        "split": "evaluation",
        "domain": "code",
        "position": 3,
        "calibration_spec_sha256": frozen,
        "expert_ids": list(range(10, 18)),
        "freeze_state": encode_state(_state(), page_cap=cap),
        "selected_state": encode_state(_state(*bits), page_cap=cap),
        "moves": _moves(*bits),
        "parameter": parameter,
        "d1_crossings": crossings,
        "d1_violation_depth": float(crossings),
        "d1_routing_mass_churn": float(crossings) / 10.0,
        "d1_labeled_margin": -float(crossings),
        "local_damage": local_damage,
        "all_q2_damage": 1.0,
    }


def _allocation_manifest_for_analysis() -> dict[str, object]:
    frozen = freeze_calibration_spec({"window": 16, "eta": 0.001})
    a, b, c, d = ((0, 0, 1),), ((1, 1, 2),), ((2, 2, 4),), ((3, 3, 1),)
    records = []
    for low_rate, high_rate in ((360, 384), (725, 749)):
        for rate, endpoint in ((low_rate, "low"), (high_rate, "high")):
            high = rate == high_rate
            values = {
                analyzer.ARM_PR13: (a + b + (d if high else ()), 1),
                analyzer.ARM_STRICT: (a + b + (d if high else ()), 1),
                analyzer.ARM_LOCAL: (a + b + (d if high else ()), 2 if not high else 1),
                analyzer.ARM_D1: (b + c + (d if high else ()), 0),
                analyzer.ARM_SEVERITY: (b + c + (d if high else ()), 0),
            }
            for arm, (bits, crossings) in values.items():
                records.append(_allocation(
                    frozen["sha256"], arm=arm, rate=rate, bits=bits,
                    crossings=crossings,
                    local_damage=(
                        1.0001
                        if arm in {analyzer.ARM_D1, analyzer.ARM_SEVERITY}
                        else 1.0
                    ),
                    endpoint=endpoint,
                ))
        reference = next(
            row for row in records
            if row["arm"] == analyzer.ARM_PR13 and row["rate"] == high_rate
        )
        for row in records:
            if (
                row["arm"] not in {
                    analyzer.ARM_LOCAL, analyzer.ARM_D1, analyzer.ARM_SEVERITY,
                }
                or row["rate"] not in {low_rate, high_rate}
            ):
                continue
            row["parameter"]["candidate"]["paired_pr13_high_core_seal"] = {
                "schema": "pr13_d1_paired_reference_high_core_seal_v1",
                "reference_arm": analyzer.ARM_PR13,
                "reference_rate": high_rate,
                "reference_state_sha256": reference["selected_state"]["sha256"],
                "reference_pages": reference["selected_state"]["page_count"],
                "common_core_state_sha256": row["freeze_state"]["sha256"],
                "common_core_pages": row["freeze_state"]["page_count"],
                "common_core_is_literal_subset": True,
                "removed_pages": (
                    reference["selected_state"]["page_count"]
                    - row["freeze_state"]["page_count"]
                ),
                "added_pages": 0,
            }
    chains = []
    for low_rate, high_rate in ((360, 384), (725, 749)):
        for arm in (analyzer.ARM_LOCAL, analyzer.ARM_D1, analyzer.ARM_SEVERITY):
            low = next(
                row for row in records if row["arm"] == arm and row["rate"] == low_rate
            )
            high = next(
                row for row in records if row["arm"] == arm and row["rate"] == high_rate
            )
            chains.append({
                "name": f"{arm}_{low_rate}_{high_rate}",
                "members": [_reference(low), _reference(high)],
            })
    return {
        "schema": ALLOCATION_MANIFEST_SCHEMA,
        "frozen_calibration": frozen,
        "allocations": records,
        "nesting_chains": chains,
    }


def test_allocation_metrics_report_repair_nesting_fallback_and_early_stop() -> None:
    allocations, comparisons, nesting = analyzer._allocation_tables(
        _allocation_manifest_for_analysis(), _config(1, (0,)),
    )
    d1_low = comparisons[
        comparisons["comparison_arm"].eq(analyzer.ARM_D1)
        & comparisons["rate_pages_per_expert"].eq(360)
    ].iloc[0]
    assert d1_low["crossings_prevented"] == 2
    assert d1_low["pages_removed_from_incumbent"] == 1
    assert d1_low["pages_added_to_incumbent"] == 1
    assert d1_low["pair_fallback_counted"]
    assert not d1_low["local_guardrail_violation"]
    assert allocations["early_stop"].all()
    assert nesting["literal_physical_subset"].all()
    assert nesting["pages_removed_low_to_high"].eq(0).all()
    audit = analyzer._nesting_grid_audit(
        _config(1, (0,)), allocations, nesting,
    )
    assert audit["expected_chains"] == 6
    assert audit["paired_pr13_high_core_seals_complete"]

    incomplete = nesting.iloc[:-1].copy()
    with pytest.raises(RuntimeError, match="exact three-arm"):
        analyzer._nesting_grid_audit(
            _config(1, (0,)), allocations, incomplete,
        )

    summary = analyzer._allocation_summary(comparisons)
    overall = summary[
        summary["stratum_type"].eq("overall")
        & summary["comparison_arm"].eq(analyzer.ARM_D1)
        & summary["rate_pages_per_expert"].eq(360)
    ].iloc[0]
    assert overall["pair_fallback_pairs"] == 1
    assert overall["early_stop_fraction"] == 1.0


def test_reference_high_core_seal_is_recomputed_not_trusted() -> None:
    manifest = _allocation_manifest_for_analysis()
    nested = next(
        row for row in manifest["allocations"]
        if row["arm"] == analyzer.ARM_D1 and row["rate"] == 360
    )
    nested["parameter"]["candidate"]["paired_pr13_high_core_seal"][
        "reference_pages"
    ] += 1
    with pytest.raises(RuntimeError, match="does not reproduce exactly"):
        analyzer._allocation_tables(manifest, _config(1, (0,)))


def test_promotion_requires_engineering_nominal_ci_holm_and_tail_gates() -> None:
    allocations, comparisons, nesting = analyzer._allocation_tables(
        _allocation_manifest_for_analysis(), _config(1, (0,)),
    )
    inference = pd.DataFrame([
        {
            "contrast": spec[0],
            "mean": -0.01,
            "bootstrap_lower": -0.02,
            "bootstrap_upper": -0.001,
            "sign_flip_p_raw": 0.001,
            "sign_flip_p_holm": 0.004,
            "p95_regression": -0.0001,
            "maximum_regression": 0.001,
            "primary_inferential_gate_pass": True,
        }
        for spec in analyzer.PRIMARY_SPECS
    ])
    promotion = analyzer._promotion_payload(
        _config(1, (0,)), inference, allocations, comparisons, nesting,
    )
    assert promotion["status"] == "passed"
    assert not promotion["experiment_b_started"]
    assert "not Holm-adjusted" in promotion["confidence_interval_rule"]
    assert "p-values" in promotion["multiplicity_rule"]
    engineering = promotion["engineering_gates"]
    assert engineering[
        "nested_d1_exact_crossings_sum_strictly_below_local_pooled"
    ]
    assert engineering[
        "nested_d1_exact_crossings_mean_strictly_below_local_pooled"
    ]
    assert engineering["nested_d1_binary_unsafe_rate_strictly_below_local_pooled"]
    assert engineering["nested_local_safe_made_unsafe_gate_pass"]

    failed = inference.copy()
    failed.loc[0, "primary_inferential_gate_pass"] = False
    assert analyzer._promotion_payload(
        _config(1, (0,)), failed, allocations, comparisons, nesting,
    )["status"] == "failed"

    binary_equal = comparisons.copy()
    is_d1 = binary_equal["comparison_arm"].eq(analyzer.ARM_D1)
    binary_equal.loc[is_d1, "candidate_d1_crossings"] = 1
    binary_equal.loc[is_d1, "candidate_d1_unsafe"] = True
    binary_equal.loc[is_d1, "safe_to_unsafe"] = False
    binary_gate = analyzer._promotion_payload(
        _config(1, (0,)), inference, allocations, binary_equal, nesting,
    )
    assert not binary_gate["engineering_gates"][
        "nested_d1_binary_unsafe_rate_strictly_below_local_pooled"
    ]
    assert binary_gate["status"] == "failed"

    unsafe_introduced = comparisons.copy()
    row = unsafe_introduced[
        unsafe_introduced["comparison_arm"].eq(analyzer.ARM_D1)
    ].index[0]
    unsafe_introduced.loc[row, "incumbent_d1_crossings"] = 0
    unsafe_introduced.loc[row, "incumbent_d1_unsafe"] = False
    unsafe_introduced.loc[row, "candidate_d1_crossings"] = 1
    unsafe_introduced.loc[row, "candidate_d1_unsafe"] = True
    unsafe_introduced.loc[row, "safe_to_unsafe"] = True
    safe_gate = analyzer._promotion_payload(
        _config(1, (0,)), inference, allocations, unsafe_introduced, nesting,
    )
    assert not safe_gate["engineering_gates"][
        "nested_local_safe_made_unsafe_gate_pass"
    ]
    assert safe_gate["status"] == "failed"
