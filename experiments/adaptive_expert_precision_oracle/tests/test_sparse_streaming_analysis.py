from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from oracle_study.sparse_streaming_analysis import (
    AnalysisValidationError,
    _markdown_table,
    analyze,
    choose_validation_promotions,
    frozen_json_sha256,
    load_and_validate_inputs,
    schema_document,
    write_frozen_json,
)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": "synthetic_sparse_streaming",
        "layers": [4],
        "primary_metric": "expert-output qenergy recovery; not task accuracy",
        "activation_shortlist_unit": "input_coordinates",
        "coordinate_score_aggregation": "sum_stage_energy",
        "shortlist_physical_layout": "canonical_paired_planes_four_coordinates_per_projection_page",
        "tile_pilot_shape": [32, 32],
        "page_size_bytes": 512,
        "reference": {
            "revision": "locked-revision",
            "config_sha256": "config-hash",
            "index_sha256": "index-hash",
        },
        "locked_tree_sha256": "tree-hash",
        "locked_capture_sha256": {
            "exact_checkpoint": "fresh-hash",
            "cross_reference": "cross-hash",
        },
        "promotion_policy": {
            "selection_split": "validation",
            "promotion_min_validation_invocations_per_layer": 2,
            "promotion_min_validation_requests_per_layer": 2,
            "shortlist_overfetch_statistic": "p90",
            "unit_go_at_1bpw_median": 0.90,
            "unit_go_at_1bpw_p10": 0.90,
            "tile_min_recovery_point_gain": 0.03,
            "tile_min_page_reduction_at_matched_recovery": 0.25,
            "shortlist_min_exact_gain_retention_median": 0.95,
            "shortlist_min_exact_gain_retention_p10": 0.90,
            "shortlist_max_candidates": 512,
            "shortlist_min_overfetch": 1.0,
            "shortlist_max_overfetch": 1.25,
            "no_test_tuning": True,
        },
        "locked_pr6_baselines_at_1_physical_bpw": {
            "h0_hybrid_representation": {"p10": 0.87, "median": 0.90}
        },
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_dir = root / "run"
    run_dir.mkdir()
    facts = {
        "run_id": "pilot-1",
        "config_sha256": config_hash,
        "reference": config["reference"],
        "locked_tree_sha256": config["locked_tree_sha256"],
        "locked_capture_sha256": config["locked_capture_sha256"],
        "page_size_bytes": 512,
        "codec_locked": True,
        "request_separation_verified": True,
        "run_mode": "pilot",
        "completed": True,
        "expected_unique_invocations": 4,
        "observed_unique_invocations": 4,
    }
    (run_dir / "run_facts.json").write_text(json.dumps(facts, sort_keys=True) + "\n")

    common_rows = []
    for split in ("validation", "test"):
        for sample in range(2):
            common_rows.append({
                "capture_source": "exact_checkpoint",
                "evaluation_split": split,
                "request_id": f"{split}-{sample}",
                "position": sample,
                "layer": 4,
                "expert_id": 154,
            })

    neuron_rows = []
    for common in common_rows:
        for objective in (
            "isolated_projection_qenergy",
            "exact_sequential_complete_expert_qenergy",
        ):
            for selector in ("validation_winner", "test_winner"):
                if common["evaluation_split"] == "validation":
                    recovery = 0.96 if selector == "validation_winner" else 0.93
                else:
                    recovery = 0.10 if selector == "validation_winner" else 0.99
                pages = 768
                logical_bytes = pages * 512
                neuron_rows.append({
                    **common,
                    "objective": objective,
                    "selection_regime": "h0_resident_proxy_late",
                    "physical_budget_bpw": 1.0,
                    "physical_pages": pages,
                    "physical_bytes": pages * 512,
                    "logical_actions": 256,
                    "logical_bytes": logical_bytes,
                    "logical_bpw": 8.0 * logical_bytes / (3 * 2048 * 512),
                    "recovery": recovery,
                    "selector_runtime_ms": 2.0 if selector == "validation_winner" else 1.0,
                    "selector_bytes_read": 4096,
                    "suffix_storage_bpw": 2.0,
                    "selector_metadata_bpw": 0.0,
                    "selector_metadata_bytes_per_expert": 0,
                    "storage_multiplier": (4.25 + 2.0) / 4.25,
                    "page_amplification": 1.0,
                    "action_family": "neuron_major",
                    "representation": "full_q4_unit_packet",
                    "selector": selector,
                })
    pd.DataFrame(neuron_rows).to_parquet(run_dir / "neuron_major_frontier.parquet", index=False)

    shortlist_rows = []
    for common in common_rows:
        for method, retained in (
            ("validation_winner", 0.97 if common["evaluation_split"] == "validation" else 0.10),
            ("test_winner", 0.93 if common["evaluation_split"] == "validation" else 0.99),
        ):
            logical_bytes = 300 * 512
            shortlist_rows.append({
                **common,
                "projection": "gate",
                "objective": "isolated_projection_qenergy",
                "selection_regime": "h0_resident_proxy_late",
                "score_method": method,
                "shortlist_size": 256,
                "refresh_block": 8,
                "candidate_overfetch_factor": 1.2,
                "physical_budget_bpw": 1.0,
                "candidate_pages": 320,
                "fetched_pages": 300,
                "physical_pages": 300,
                "physical_bytes": 300 * 512,
                "logical_actions": 256,
                "logical_bytes": logical_bytes,
                "logical_bpw": 8.0 * logical_bytes / (2048 * 512),
                "recovery": 0.9,
                "exact_gain_retained": retained,
                "exact_action_recall": 0.6,
                "importance_weighted_recall": 0.9,
                "recovery_diagonal": 0.7,
                "recovery_exact": 0.95,
                "recovery_constrained": 0.9,
                "recovery_exact_matched_fetched_pages": 0.95,
                "recovery_diagonal_matched_fetched_pages": 0.7,
                "fraction_exact_over_diagonal_gain_retained_matched_fetched_pages": retained,
                "selector_runtime_ms": 1.0,
                "selector_bytes_read": 2048,
                "suffix_storage_bpw": 2.0,
                "selector_metadata_bpw": 0.0,
                "selector_metadata_bytes_per_expert": 0,
                "storage_multiplier": (4.25 + 2.0) / 4.25,
                "page_amplification": 1.0,
            })
    pd.DataFrame(shortlist_rows).to_parquet(run_dir / "activation_shortlist_containment.parquet", index=False)

    label_rows = []
    for common in common_rows:
        for rank in (0, 1):
            label_rows.append({
                **common,
                "projection": "gate",
                "action_rank": rank,
                "action_id": rank,
                "coordinate": rank,
                "refinement_stage": 1,
                "marginal_gain": 1.0 - 0.1 * rank,
                "physical_page_id": 0,
            })
    pd.DataFrame(label_rows).to_parquet(run_dir / "exact_action_labels.parquet", index=False)

    tile_rows = []
    for common in common_rows:
        for budget, pages in ((0.5, 384), (1.0, 768)):
            baseline_recovery = 0.75 if budget == 0.5 else 0.90
            for family, shape, selector in (
                ("input_coordinate_baseline", "none", "pr6_control"),
                ("tile", "32x32", "validation_winner"),
                ("tile", "64x16", "test_winner"),
            ):
                if family == "input_coordinate_baseline":
                    recovery = baseline_recovery
                elif common["evaluation_split"] == "validation":
                    recovery = baseline_recovery + (0.05 if selector == "validation_winner" else 0.01)
                else:
                    recovery = 0.10 if selector == "validation_winner" else 0.99
                logical_bytes = pages * 512
                tile_rows.append({
                    **common,
                    "objective": "exact_sequential_complete_expert_qenergy",
                    "selection_regime": "h0_resident_proxy_late",
                    "physical_budget_bpw": budget,
                    "physical_pages": pages,
                    "physical_bytes": pages * 512,
                    "logical_actions": pages,
                    "logical_bytes": logical_bytes,
                    "logical_bpw": 8.0 * logical_bytes / (3 * 2048 * 512),
                    "recovery": recovery,
                    "selector_runtime_ms": 3.0,
                    "selector_bytes_read": 8192,
                    "suffix_storage_bpw": 2.0,
                    "selector_metadata_bpw": 0.0,
                    "selector_metadata_bytes_per_expert": 0,
                    "storage_multiplier": (4.25 + 2.0) / 4.25,
                    "page_amplification": 1.0,
                    "action_family": family,
                    "tile_shape": shape,
                    "selector": selector,
                })
    pd.DataFrame(tile_rows).to_parquet(run_dir / "tile_streaming_frontier.parquet", index=False)

    stability_rows = []
    for common in common_rows:
        stability_rows.append({
            **common,
            "projection": "gate",
            "score_method": "abs_activation",
            "shortlist_size": 256,
            "token_gap": 1,
            "relationship": "adjacent_token_same_sequence",
            "support_jaccard": 0.7,
            "importance_weighted_overlap": 0.9,
        })
    pd.DataFrame(stability_rows).to_parquet(run_dir / "support_stability.parquet", index=False)
    concentration_rows = []
    for common in common_rows:
        concentration_rows.append({
            **common,
            "top_k": 512,
            "top_k_energy_fraction": 0.81,
            "exact_zero_fraction": 0.001,
            "coordinates_for_90pct_energy": 762,
        })
    pd.DataFrame(concentration_rows).to_parquet(
        run_dir / "activation_concentration.parquet", index=False
    )
    return run_dir, config_path


def test_schema_document_names_every_required_artifact() -> None:
    schema = schema_document()
    assert set(schema["input_files"].values()) == {
        "neuron_major_frontier.parquet",
        "activation_shortlist_containment.parquet",
        "exact_action_labels.parquet",
        "tile_streaming_frontier.parquet",
        "support_stability.parquet",
        "activation_concentration.parquet",
    }


def test_internal_markdown_renderer_does_not_require_tabulate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_to_markdown(*args, **kwargs):
        raise ModuleNotFoundError("tabulate intentionally unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", forbidden_to_markdown)
    rendered = _markdown_table(
        pd.DataFrame({"method": ["gate|up", "down\ncolumn"], "recovery": [0.9, 1.0]}),
        ["method", "recovery"],
    )
    assert rendered == (
        "| method | recovery |\n"
        "| --- | --- |\n"
        "| gate\\|up | 0.9000 |\n"
        "| down<br>column | 1.0000 |"
    )


def test_promotion_is_validation_only_and_outputs_are_complete(tmp_path: Path) -> None:
    run_dir, config_path = _write_fixture(tmp_path)
    output = tmp_path / "analysis"
    analyze([run_dir], config_path, output)
    promotions = json.loads((output / "sparse_streaming_promotions.json").read_text())
    assert promotions["test_rows_consulted_for_selection"] is False
    assert promotions["neuron_major"]["promoted"][0]["selector"] == "validation_winner"
    assert promotions["activation_shortlist"]["promoted"][0]["score_method"] == "validation_winner"
    assert promotions["tile_streaming"]["promoted"][0]["selector"] == "validation_winner"
    report = (output / "SPARSE_STREAMING_ALLOCATOR_REPORT.md").read_text()
    assert "expert-output qenergy recovery" in report
    assert "not task accuracy" in report
    assert "Logical actions" in report
    for name in (
        "neuron_major_summary.csv",
        "neuron_major_by_layer_summary.csv",
        "activation_shortlist_summary.csv",
        "activation_shortlist_by_layer_summary.csv",
        "tile_streaming_summary.csv",
        "tile_streaming_by_layer_summary.csv",
        "support_stability_summary.csv",
        "exact_action_label_summary.csv",
        "activation_concentration_summary.csv",
        "sparse_streaming_compute_storage_accounting.json",
        "analysis_manifest.json",
    ):
        assert (output / name).is_file()
    neuron_by_layer = pd.read_csv(output / "neuron_major_by_layer_summary.csv")
    assert set(neuron_by_layer.layer) == {4}
    assert {"n", "p10_recovery", "median_recovery", "p90_recovery"} <= set(
        neuron_by_layer.columns
    )
    assert len(list((output / "plots").glob("*.png"))) == 6
    assert len(list((output / "plots").glob("*.svg"))) == 6


def test_validate_only_writes_nothing(tmp_path: Path) -> None:
    run_dir, config_path = _write_fixture(tmp_path)
    output = tmp_path / "must_not_exist"
    result = analyze([run_dir], config_path, output, validate_only=True)
    assert result["validated"] is True
    assert not output.exists()


def test_missing_schema_column_fails_loudly(tmp_path: Path) -> None:
    run_dir, config_path = _write_fixture(tmp_path)
    path = run_dir / "neuron_major_frontier.parquet"
    pd.read_parquet(path).drop(columns="physical_pages").to_parquet(path, index=False)
    with pytest.raises(AnalysisValidationError, match="physical_pages"):
        load_and_validate_inputs([run_dir], config_path)


def test_logical_accounting_is_required_and_recomputed(tmp_path: Path) -> None:
    run_dir, config_path = _write_fixture(tmp_path)
    path = run_dir / "neuron_major_frontier.parquet"
    values = pd.read_parquet(path)
    values.drop(columns="logical_bytes").to_parquet(path, index=False)
    with pytest.raises(AnalysisValidationError, match="logical_bytes"):
        load_and_validate_inputs([run_dir], config_path)

    run_dir, config_path = _write_fixture(tmp_path / "wrong_bpw")
    path = run_dir / "tile_streaming_frontier.parquet"
    values = pd.read_parquet(path)
    values["logical_bpw"] += 0.01
    values.to_parquet(path, index=False)
    with pytest.raises(AnalysisValidationError, match="logical_bpw must equal"):
        load_and_validate_inputs([run_dir], config_path)


def test_shortlist_fetched_pages_may_exceed_applied_logical_payload(tmp_path: Path) -> None:
    run_dir, config_path = _write_fixture(tmp_path)
    path = run_dir / "activation_shortlist_containment.parquet"
    values = pd.read_parquet(path)
    values["logical_bytes"] = 100 * 512
    values["logical_bpw"] = 8.0 * values["logical_bytes"] / (2048 * 512)
    values["page_amplification"] = values["physical_bytes"] / values["logical_bytes"]
    values.to_parquet(path, index=False)
    load_and_validate_inputs([run_dir], config_path)

    values["page_amplification"] = 1.0
    values.to_parquet(path, index=False)
    with pytest.raises(AnalysisValidationError, match="page_amplification must equal"):
        load_and_validate_inputs([run_dir], config_path)


def test_frozen_promotions_refuse_changed_decision(tmp_path: Path) -> None:
    path = tmp_path / "promotions.json"
    write_frozen_json(path, {"selector": "validation_winner"})
    write_frozen_json(path, {"selector": "validation_winner"})
    with pytest.raises(AnalysisValidationError, match="refusing to overwrite"):
        write_frozen_json(path, {"selector": "test_winner"})


def test_promotion_fails_when_a_frozen_layer_is_missing(tmp_path: Path) -> None:
    run_dir, config_path = _write_fixture(tmp_path)
    tables, _, config, hashes = load_and_validate_inputs([run_dir], config_path)
    config["layers"] = [4, 20]
    with pytest.raises(AnalysisValidationError, match="lacks frozen validation coverage"):
        choose_validation_promotions(tables, config, "synthetic", hashes)


def test_shortlist_underfetch_cannot_pass_matched_page_gate(tmp_path: Path) -> None:
    run_dir, config_path = _write_fixture(tmp_path)
    path = run_dir / "activation_shortlist_containment.parquet"
    values = pd.read_parquet(path)
    values.loc[values.evaluation_split == "validation", "candidate_overfetch_factor"] = 0.75
    values.to_parquet(path, index=False)
    tables, _, config, hashes = load_and_validate_inputs([run_dir], config_path)
    promotions = choose_validation_promotions(tables, config, "synthetic", hashes)
    shortlist = promotions["activation_shortlist"]
    assert shortlist["status"] == "stop"
    assert shortlist["promoted"] == []
    candidate = next(
        row for row in shortlist["all_validation_candidates"]
        if row["score_method"] == "validation_winner"
    )
    assert candidate["p10_candidate_overfetch"] == pytest.approx(0.75)
    assert candidate["passes_min_overfetch"] is False


def test_storage_multiplier_uses_reference_bpw_denominator(tmp_path: Path) -> None:
    run_dir, config_path = _write_fixture(tmp_path)
    path = run_dir / "neuron_major_frontier.parquet"
    values = pd.read_parquet(path)
    values["storage_multiplier"] = (2.25 + 2.0) / 2.25
    values.to_parquet(path, index=False)
    with pytest.raises(AnalysisValidationError, match="resident-parent bpw is not the denominator"):
        load_and_validate_inputs([run_dir], config_path)


def _copy_as_full(run_dir: Path, destination: Path) -> Path:
    shutil.copytree(run_dir, destination)
    facts_path = destination / "run_facts.json"
    facts = json.loads(facts_path.read_text())
    facts["run_id"] = "full-1"
    facts["run_mode"] = "full"
    facts_path.write_text(json.dumps(facts, sort_keys=True) + "\n")
    return destination


def _expected_promotion_hash(run_dir: Path, config_path: Path) -> str:
    tables, _, config, hashes = load_and_validate_inputs([run_dir], config_path)
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    promotions = choose_validation_promotions(tables, config, config_digest, hashes)
    return frozen_json_sha256(promotions)


def _set_full_promotion_hash(run_dir: Path, digest: str) -> None:
    facts_path = run_dir / "run_facts.json"
    facts = json.loads(facts_path.read_text())
    facts["validation_promotions_sha256"] = digest
    facts_path.write_text(json.dumps(facts, sort_keys=True) + "\n")


def test_full_run_promotion_hash_matches_validation_payload(tmp_path: Path) -> None:
    pilot, config_path = _write_fixture(tmp_path)
    expected = _expected_promotion_hash(pilot, config_path)
    full = _copy_as_full(pilot, tmp_path / "full")
    _set_full_promotion_hash(full, "0" * 64)
    with pytest.raises(AnalysisValidationError, match="does not match"):
        analyze([pilot, full], config_path, tmp_path / "bad", validate_only=True)

    _set_full_promotion_hash(full, expected)
    result = analyze([pilot, full], config_path, tmp_path / "good", validate_only=True)
    assert result["validation_promotions_sha256"] == expected


def test_all_full_runs_share_one_promotion_hash(tmp_path: Path) -> None:
    pilot, config_path = _write_fixture(tmp_path)
    expected = _expected_promotion_hash(pilot, config_path)
    full_a = _copy_as_full(pilot, tmp_path / "full_a")
    full_b = _copy_as_full(pilot, tmp_path / "full_b")
    _set_full_promotion_hash(full_a, expected)
    _set_full_promotion_hash(full_b, "1" * 64)
    with pytest.raises(AnalysisValidationError, match="same frozen validation promotion"):
        analyze(
            [pilot, full_a, full_b], config_path, tmp_path / "bad_pair",
            validate_only=True,
        )


def test_full_rows_replace_agreeing_pilot_duplicates(tmp_path: Path) -> None:
    pilot, config_path = _write_fixture(tmp_path)
    pilot_tables, _, pilot_config, pilot_hashes = load_and_validate_inputs([pilot], config_path)
    pilot_promotions = choose_validation_promotions(
        pilot_tables, pilot_config, "synthetic", pilot_hashes
    )
    full = _copy_as_full(pilot, tmp_path / "full")
    tables, _, config, merged_hashes = load_and_validate_inputs([pilot, full], config_path)
    merged_promotions = choose_validation_promotions(
        tables, config, "synthetic", merged_hashes
    )
    assert merged_promotions == pilot_promotions
    assert set(tables["neuron"].source_run_mode) == {"full"}
    stats = config["_analysis_deduplication"]["neuron"]
    assert stats["pre_rows"] == 2 * stats["post_rows"]
    assert stats["rows_removed"] == stats["post_rows"]


def test_duplicate_recovery_disagreement_fails_loudly(tmp_path: Path) -> None:
    pilot, config_path = _write_fixture(tmp_path)
    full = _copy_as_full(pilot, tmp_path / "full")
    path = full / "neuron_major_frontier.parquet"
    values = pd.read_parquet(path)
    values.loc[0, "recovery"] += 0.01
    values.to_parquet(path, index=False)
    with pytest.raises(AnalysisValidationError, match="disagrees in evidence column 'recovery'"):
        load_and_validate_inputs([pilot, full], config_path)
