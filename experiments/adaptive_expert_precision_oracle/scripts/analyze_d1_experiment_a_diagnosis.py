#!/usr/bin/env python3
"""Build the immutable, GPU-free D1 Experiment A diagnosis package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_decode_tail import (  # noqa: E402
    apply_same_host_allocation_patch,
    assemble_decode_policy_bank,
    expected_grid_counts,
    load_decode_oracle_cell,
    sha256,
    validate_cached_decode_tail_config,
)
from oracle_study.d1_experiment_a_diagnosis import (  # noqa: E402
    RATE_PAIRS,
    SCHEMA,
    adjacent_rate_nonmonotonicity,
    crossing_conditional_effects,
    deployable_d1_target_remedies,
    heavy_tail_summaries,
    source_adjacent_rate_geometry,
)


TAIL_SCHEMA = "pr13_d1_cached_decode_tail_kl_v2"
QUALITY = "d1_cached_decode_tail_quality.parquet"
PROPAGATION = "d1_cached_decode_tail_propagation.parquet"
ZERO = "d1_cached_decode_tail_zero_gates.parquet"
RUN_FACTS = "d1_cached_decode_tail_run_facts.json"

OUTPUTS = {
    "crossing_rows": "d1_experiment_a_crossing_conditional_rows.parquet",
    "crossing_summary": "d1_experiment_a_crossing_conditional_summary.parquet",
    "adjacent_rows": "d1_experiment_a_adjacent_rate_nonmonotonicity_rows.parquet",
    "adjacent_summary": "d1_experiment_a_adjacent_rate_nonmonotonicity_summary.parquet",
    "heavy_tail_summary": "d1_experiment_a_heavy_tail_summary.parquet",
    "heavy_tail_events": "d1_experiment_a_heavy_tail_events.parquet",
    "remedy_rows": "d1_experiment_a_deployable_target_remedy_rows.parquet",
    "remedy_summary": "d1_experiment_a_deployable_target_remedy_summary.parquet",
    "source_experts": "d1_experiment_a_source_adjacent_expert_geometry.parquet",
    "source_deltas": "d1_experiment_a_source_adjacent_delta_geometry.parquet",
    "source_summary": "d1_experiment_a_source_adjacent_geometry_summary.parquet",
}
FACTS = "d1_experiment_a_diagnosis_facts.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _verified_tail_inputs(
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    facts_path = root / RUN_FACTS
    facts = _load_json(facts_path)
    if facts.get("schema") != TAIL_SCHEMA or not bool(facts.get("completed")):
        raise RuntimeError("cached-decode tail run is not finalized")
    if str(facts.get("config_sha256")) != sha256(config_path):
        raise RuntimeError("cached-decode tail config hash changed")
    frames: dict[str, pd.DataFrame] = {}
    for name in (QUALITY, PROPAGATION, ZERO):
        path = root / name
        recorded = facts.get("outputs", {}).get(name, {})
        if not path.is_file() or str(recorded.get("sha256")) != sha256(path):
            raise RuntimeError(f"cached-decode tail table changed: {name}")
        frame = pd.read_parquet(path)
        if len(frame) != int(recorded.get("rows", -1)):
            raise RuntimeError(f"cached-decode tail row count changed: {name}")
        frames[name] = frame
    counts = expected_grid_counts(config)
    expected = {
        QUALITY: counts["quality_rows"],
        PROPAGATION: counts["propagation_rows"],
        ZERO: counts["logical_zero_dose_rows"],
    }
    for name, rows in expected.items():
        if len(frames[name]) != int(rows):
            raise RuntimeError(f"cached-decode tail grid is incomplete: {name}")
    if not bool(frames[ZERO]["all_exact"].all()):
        raise RuntimeError("zero-dose parity failed")
    return frames[QUALITY], frames[PROPAGATION], frames[ZERO], facts


def _cell_directory(root: Path, rate: int, layer: int) -> Path:
    return root / f"rate_{int(rate)}_layer{int(layer):02d}" / f"layer_{int(layer):02d}"


def _load_policy_banks(
    config: Mapping[str, Any],
    mechanism_results: Path,
    matched_results: Path,
    patch_root: Path,
) -> tuple[dict[tuple[int, int], Any], list[dict[str, Any]], dict[str, Any]]:
    banks: dict[tuple[int, int], Any] = {}
    source_records: list[dict[str, Any]] = []
    mechanism_rates = set(map(int, config["mechanism_page_caps"]))
    matched_rates = set(map(int, config["matched_runtime_metadata_page_caps"]))
    for layer in map(int, config["injection_layers"]):
        for rate in map(int, config["page_caps"]):
            if rate in mechanism_rates:
                root = mechanism_results
                source_kind = "mechanism"
            elif rate in matched_rates:
                root = matched_results
                source_kind = "matched_runtime_metadata"
            else:
                raise RuntimeError(f"unclassified source page cap: {rate}")
            directory = _cell_directory(root, rate, layer)
            cell = load_decode_oracle_cell(directory)
            if (int(cell.layer), int(cell.rate)) != (layer, rate):
                raise RuntimeError(f"source cell identity changed: {directory}")
            mapping = config["d1_source_policy_by_rate"][str(rate)]
            banks[(layer, rate)] = assemble_decode_policy_bank(
                cell,
                fixed_source_policy=str(mapping["fixed"]),
                companion_source_policy=str(mapping["companion"]),
            )
            facts_path = directory / "d1_decode_layer_facts.json"
            source_records.append(
                {
                    "layer": layer,
                    "rate": rate,
                    "source_kind": source_kind,
                    "directory": str(directory),
                    "facts_sha256": sha256(facts_path),
                    "recorded_files": {
                        name: {
                            "sha256": str(record["sha256"]),
                            "bytes": int(record["bytes"]),
                        }
                        for name, record in sorted(cell.facts["files"].items())
                    },
                }
            )
    patch_config = config["same_host_allocation_patch"]
    banks = apply_same_host_allocation_patch(
        banks,
        patch_root,
        expected_facts_sha256=str(patch_config["facts_sha256"]),
        expected_mismatched_groups=int(patch_config["mismatched_groups"]),
    )
    patch_facts_path = patch_root / "d1_tail_same_host_patch_facts.json"
    patch_facts = _load_json(patch_facts_path)
    patch_record = {
        "directory": str(patch_root),
        "facts_sha256": sha256(patch_facts_path),
        "mismatched_groups": int(patch_facts["mismatched_groups"]),
        "outputs": {
            name: {
                "sha256": str(record["sha256"]),
                "bytes": int(record["bytes"]),
            }
            for name, record in sorted(patch_facts["outputs"].items())
        },
    }
    return banks, source_records, patch_record


def _arguments() -> argparse.Namespace:
    result_root = (
        EXPERIMENT / "results/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT
        / "configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3.json",
    )
    parser.add_argument("--input", type=Path, default=result_root)
    parser.add_argument(
        "--mechanism-results",
        type=Path,
        default=EXPERIMENT
        / "results/qwen36_mxfp4_d1_exact_prefill_decode_slice_expansion_20260824_v2/results",
    )
    parser.add_argument(
        "--matched-results",
        type=Path,
        default=EXPERIMENT
        / "results/qwen36_mxfp4_d1_decode_matched_rates_20260824_v2/results",
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        default=result_root / "same_host_allocation_patch",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=result_root / "diagnosis",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config_path = args.config.resolve()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    config = _load_json(config_path)
    validate_cached_decode_tail_config(config)
    quality, propagation, zero, run_facts = _verified_tail_inputs(
        input_root, config, config_path
    )
    banks, source_records, patch_record = _load_policy_banks(
        config,
        args.mechanism_results.resolve(),
        args.matched_results.resolve(),
        args.patch_root.resolve(),
    )

    crossing_rows, crossing_summary = crossing_conditional_effects(quality, propagation)
    adjacent_rows, adjacent_summary = adjacent_rate_nonmonotonicity(quality, RATE_PAIRS)
    heavy_tail_summary, heavy_tail_events = heavy_tail_summaries(
        crossing_rows, adjacent_rows
    )
    remedy_rows, remedy_summary = deployable_d1_target_remedies(
        quality, propagation, RATE_PAIRS
    )
    source_experts, source_deltas, source_summary = source_adjacent_rate_geometry(
        banks, RATE_PAIRS
    )
    frames = {
        OUTPUTS["crossing_rows"]: crossing_rows,
        OUTPUTS["crossing_summary"]: crossing_summary,
        OUTPUTS["adjacent_rows"]: adjacent_rows,
        OUTPUTS["adjacent_summary"]: adjacent_summary,
        OUTPUTS["heavy_tail_summary"]: heavy_tail_summary,
        OUTPUTS["heavy_tail_events"]: heavy_tail_events,
        OUTPUTS["remedy_rows"]: remedy_rows,
        OUTPUTS["remedy_summary"]: remedy_summary,
        OUTPUTS["source_experts"]: source_experts,
        OUTPUTS["source_deltas"]: source_deltas,
        OUTPUTS["source_summary"]: source_summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        _atomic_parquet(output_root / name, frame)
    output_records = {
        name: {
            "rows": int(len(frame)),
            "bytes": int((output_root / name).stat().st_size),
            "sha256": sha256(output_root / name),
        }
        for name, frame in sorted(frames.items())
    }
    facts = {
        "schema": SCHEMA,
        "completed": True,
        "offline_only": True,
        "gpu_work_performed": False,
        "test_rows_admitted_or_used": False,
        "terminal_kl_used_for_remedy_selection": False,
        "implementation": {
            "scripts/analyze_d1_experiment_a_diagnosis.py": sha256(
                Path(__file__).resolve()
            ),
            "src/oracle_study/d1_experiment_a_diagnosis.py": sha256(
                EXPERIMENT / "src/oracle_study/d1_experiment_a_diagnosis.py"
            ),
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256(config_path),
        },
        "tail_inputs": {
            "root": str(input_root),
            "run_facts_sha256": sha256(input_root / RUN_FACTS),
            "run_schema": str(run_facts["schema"]),
            "quality": run_facts["outputs"][QUALITY],
            "propagation": run_facts["outputs"][PROPAGATION],
            "zero_gates": run_facts["outputs"][ZERO],
            "zero_rows": int(len(zero)),
            "zero_dose_all_exact": True,
        },
        "source_cells": source_records,
        "same_host_allocation_patch": patch_record,
        "algorithm": {
            "rate_pairs": [list(pair) for pair in RATE_PAIRS],
            "crossing_events": [
                "prevented",
                "introduced",
                "both_crossed",
                "both_safe",
            ],
            "heavy_tail_top_events_per_comparison": 10,
            "downstream_routes_role": "descriptive_outcomes_only_not_selection_inputs",
            "remedy_selectors": {
                "strict_pr13_incumbent_repair": {
                    "safe_pr13_is_immutable": True,
                    "unsafe_switch_requires_strictly_fewer_exact_full_model_d1_crossings": True,
                    "tie_break_order": [
                        "lowest_exact_full_model_d1_mass_churn",
                        "lowest_live_execution_weighted_local_qenergy",
                        "smallest_absolute_total_selected_group_page_count_difference",
                        "deterministic_policy_name",
                    ],
                },
                "global_hard_d1_local": {
                    "candidate_pool_includes_pr13": True,
                    "selection_order": [
                        "fewest_exact_full_model_d1_crossings",
                        "lowest_exact_full_model_d1_mass_churn",
                        "lowest_live_execution_weighted_local_qenergy",
                        "smallest_absolute_total_selected_group_page_count_difference",
                        "deterministic_policy_name",
                    ],
                },
                "adjacent_rate_hard_d1_early_stop": {
                    "candidate_pool": "five_executed_policies_at_adjacent_low_and_high_rates",
                    "hard_constraint": "selected_group_pages_le_high_rate_pr13_pages",
                },
            },
            "source_policy_banks_include_same_host_patch": True,
            "precision_state_width": 512,
            "projection_bits_per_state_unit": 3,
        },
        "outputs": output_records,
        "scientific_boundary": (
            "Deterministic post-hoc diagnosis of the finalized three-request "
            "exact-prefix cached single-token Experiment A smoke. Exact D1 "
            "route labels are oracle inputs to the remedy ablations; routes "
            "after D1 and terminal KL are descriptive outcomes only. This is "
            "not a runtime controller, generated rollout, or quality claim."
        ),
    }
    _atomic_json(output_root / FACTS, facts)
    print(
        json.dumps(
            {"facts": str(output_root / FACTS), "outputs": output_records}, indent=2
        )
    )


if __name__ == "__main__":
    main()
