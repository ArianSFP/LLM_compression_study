#!/usr/bin/env python3
"""Bounded validation-only coherent-unit set-utility study.

This runner is deliberately stacked on the PR #9 experiment rather than
replacing it.  It keeps the checkpoint, embedded Q2->Q3->Q4 hierarchy,
selected trees, and request labels immutable.  The fit phase admits only
training rows from the two locked captures.  The evaluate phase admits exact
checkpoint train rows (solely to reconstruct the frozen future-proxy metric)
and exact-checkpoint validation rows; it has no test mode.

The fit bundle contains compact coherent-unit teachers, per-expert support
templates, one bounded block-PQ residual synopsis, and high-rank/low-bit
linear controls.  It never serializes the [unit, output] correction vectors.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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

from oracle_study.direct_set_predictor import (  # noqa: E402
    CompactCoherentUnitPredictor,
    DirectSetTrainingConfig,
    DirectSetTrainingData,
    PredictorOutput,
    evaluate_candidate_reranking,
    train_direct_set_predictor,
)
from oracle_study.neuron_selector import (  # noqa: E402
    complete_unit_scores,
    descending_order,
    encode_unit_score_metadata,
    factorized_unit_outputs,
    unit_score_metadata,
)
from oracle_study.set_utility_oracles import (  # noqa: E402
    STATE_GD,
    build_nested_template_bank,
    charged_top1_template_candidates,
    charged_top2_union_candidates,
    exact_hybrid_unit_base_gram_greedy,
    exact_hybrid_unit_fixed_greedy,
    exact_monotone_unit_local_search,
)
from oracle_study.set_utility_selector import (  # noqa: E402
    EncodedBlockResidualPQ,
    QuantizedRows,
    decode_block_residual_pq,
    encode_block_residual_pq,
    evaluate_block_pq_response_lut,
    fit_activation_weighted_block_pq,
    quantize_low_bit_rows,
    decode_low_bit_rows,
)
from oracle_study.sparse_streaming import silu  # noqa: E402
from oracle_study.unit_set_teacher import (  # noqa: E402
    down_metric_gram,
    exact_contained_target_fixed_greedy,
    exact_full_target_fixed_greedy,
    set_damage,
    set_gain,
    unit_correction_gram,
)
from run_mxfp4_selective_pages import occurrence, tree_from_record  # noqa: E402
import run_neuron_selector_distillation as prior  # noqa: E402
import run_sparse_streaming_study as base  # noqa: E402


PROJECTIONS = ("gate", "up", "down")
PAGE_BYTES = 512
EXPERT_WEIGHTS = 3 * 2048 * 512
UNITS = 512
INPUTS = 2048
UNIT_PACKET_BYTES = 3 * PAGE_BYTES
PAGES_PER_BPW = EXPERT_WEIGHTS // (8 * PAGE_BYTES)
FP16_BYTES = 2
FP32_BYTES = 4
ACTIVATION_PAYLOAD_BYTES = INPUTS * FP16_BYTES
Q2_UNIT_FEATURE_BYTES = 3 * UNITS * FP16_BYTES
ABC_SCORE_MACS_PER_UNIT = 6
Q2_PARENT_CODE_BITS_PER_WEIGHT = 2
MXFP4_BLOCK_SIZE = 32
Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT = (
    2 * INPUTS * Q2_PARENT_CODE_BITS_PER_WEIGHT // 8
)
Q2_GATE_UP_SCALE_BYTES_PER_UNIT = 2 * INPUTS // MXFP4_BLOCK_SIZE
Q4_RESPONSE_SCALE_MULTIPLICATIONS_PER_UNIT = 2 * INPUTS // MXFP4_BLOCK_SIZE
ABC_SCORE_MAC_CONVENTION = (
    "six_scalar_multiplications_per_unit_for_"
    "A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_"
    "into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs"
)
RERANK_NON_MAC_CONVENTION = (
    "SiLU,_elementwise_hidden_products,_scalar_additions,_clamp,_deterministic_"
    "top-k/sort,_Q2-parent/suffix_table_decode,_E8M0_decode,_reported_Q4-response_"
    "block-scale_multiplications,_and_fixed-greedy_control_are_excluded_from_the_"
    "MAC_metric"
)
CANDIDATE_PACKET_CONTENTS = (
    "1536-byte/unit_locked_Q2-to-Q4_suffix_only:_gate/up=1024_bytes,_down=512_bytes;_"
    "excludes_resident_Q2_weights/scales,_ABC,_activation,_and_future-proxy_metadata"
)
INTERACTION_RERANK_ACCOUNTING_BOUND = (
    "precise_lower_bound_excludes_resident_Q2-down_parent-code/scale_payload,_"
    "Q2-down_tree/E8M0_decode_and_block-scale_application,_fixed-greedy_initial_"
    "correlation,_and_per-selection_vector_updates"
)
INTERACTION_RERANK_UNACCOUNTED = (
    "resident_Q2-down_parent-code/scale_payload,_tree/E8M0_decode,_and_block-scale_"
    "application;_fixed-greedy_initial_correlation_and_per-selection_vector_updates;_"
    "operations_listed_in_rerank_non_mac_operations"
)
IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer", "expert_id",
)

ARTIFACTS = {
    "hybrid_oracle_frontier": "hybrid_oracle_frontier.parquet",
    "template_frontier": "support_template_frontier.parquet",
    "selector_frontier": "pq_high_rank_selector_frontier.parquet",
    "candidate_frontier": "candidate_set_rerank_frontier.parquet",
}
ARTIFACT_SCHEMAS = {
    "hybrid_oracle_frontier": [*IDENTITY, "selector_family", "physical_budget_bpw", "recovery"],
    "template_frontier": [*IDENTITY, "template_cohort", "template_count", "candidate_units"],
    "selector_frontier": [
        *IDENTITY, "selector_config_id", "selector_family", "recovery",
        "selector_abc_score_units",
        "selector_abc_score_compute_macs",
        "selector_abc_score_macs_per_unit",
        "selector_abc_score_mac_convention",
    ],
    "candidate_frontier": [
        *IDENTITY, "selector_config_id", "rerank_semantics", "recovery",
        "rerank_q2_gate_up_parent_payload_required",
        "rerank_q2_gate_up_parent_payload_accounted",
        "rerank_q2_gate_up_parent_code_bytes_read",
        "rerank_q2_gate_up_scale_bytes_read",
        "rerank_q2_gate_up_scale_payload_already_read",
        "rerank_q4_response_scale_multiplications",
        "selector_abc_score_units", "selector_abc_score_compute_macs",
        "selector_abc_score_macs_per_unit",
        "selector_abc_score_mac_convention",
    ],
}
ARTIFACT_SCHEMAS_BY_STEM = {
    Path(filename).stem: ARTIFACT_SCHEMAS[name]
    for name, filename in ARTIFACTS.items()
}

FIT_COHORTS = (
    "routed_exact_train_primary",
    "exact_train_all_x_augmentation",
    "combined_exact_cross_train_all_x_augmentation",
)

JOINT_CANDIDATE_COVERAGE_SEMANTICS = (
    "hard_forward_top192_apply_head_union_top64_candidate_head_excluding_apply_"
    "supervised_for_teacher_top192_coverage"
)
INDEPENDENT_ABC_SCORE_SEMANTICS = (
    "fp16_rounded_activation_g2_u2_h2__exact_q4_gate_up_residual_responses__"
    "decoded_fp16_abc__full_vector_reference_only_fetched_units_charged"
)
RUNTIME_PROVENANCE_FIELDS = (
    "device",
    "gpu_names_used",
    "torch_version",
    "torch_cuda_runtime",
    "numpy_version",
    "python_version",
    "host",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_provenance(device: torch.device | str) -> dict[str, Any]:
    normalized = torch.device(device)
    if normalized.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        gpu_names = [str(torch.cuda.get_device_name(normalized))]
    else:
        gpu_names = []
    return {
        "device": str(normalized),
        "gpu_names_used": gpu_names,
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "numpy_version": str(np.__version__),
        "python_version": platform.python_version(),
        "host": platform.node(),
    }


def require_runtime_provenance_match(
    recorded: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(recorded, Mapping):
        raise RuntimeError(f"{label} runtime provenance is absent")
    missing = [field for field in RUNTIME_PROVENANCE_FIELDS if field not in recorded]
    if missing:
        raise RuntimeError(f"{label} runtime provenance lacks {missing}")
    for field in RUNTIME_PROVENANCE_FIELDS:
        if recorded[field] != current.get(field):
            raise RuntimeError(
                f"{label} runtime provenance mismatch for {field}: "
                f"{recorded[field]!r} != {current.get(field)!r}"
            )


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame = (
        pd.DataFrame(list(rows))
        if rows else pd.DataFrame(columns=ARTIFACT_SCHEMAS_BY_STEM.get(path.stem, []))
    )
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def validate_study_contract(config: Mapping[str, Any]) -> None:
    """Fail closed if a pilot can admit test rows or alter physical constants."""
    if config.get("selection_split") != "validation":
        raise RuntimeError("set-utility pilot selection must be validation-only")
    if config.get("evaluation_splits") != ["validation"]:
        raise RuntimeError("set-utility pilot must expose only exact validation evaluation")
    if config.get("fit_splits") != ["train"]:
        raise RuntimeError("set-utility fitting must expose only train rows")
    if config.get("no_test_rows_admitted") is not True or config.get("no_test_rows_used") is not True:
        raise RuntimeError("set-utility pilot must prohibit admitting or using test rows")
    if config.get("no_test_tuning") is not True:
        raise RuntimeError("set-utility pilot must prohibit test tuning")
    if int(config["page_size_bytes"]) != PAGE_BYTES or int(config["expert_weights"]) != EXPERT_WEIGHTS:
        raise RuntimeError("physical accounting constants changed")
    if int(config["pq_codebook_size"]) != 16 or int(config["pq_block_size"]) != 32:
        raise RuntimeError("bounded block-PQ geometry must remain 16 codewords over 32-weight blocks")
    if int(config["candidate_units"]) != 256 or int(config["applied_units"]) != 192:
        raise RuntimeError("bounded system interface must fetch 256 and apply 192 coherent units")
    cohorts = tuple(map(str, config["template_training_cohorts"]))
    if cohorts != FIT_COHORTS:
        raise RuntimeError("template cohorts must preserve routed and synthetic provenance")
    if int(config["teacher_path_length"]) != UNITS:
        raise RuntimeError("teacher path must serialize a full unit permutation")
    hybrid_cap = int(config["max_hybrid_oracle_invocations_per_expert"])
    if hybrid_cap != int(config["max_validation_invocations_per_expert"]):
        raise RuntimeError("hybrid oracle must cover every admitted validation invocation")
    if float(config["direct_predictor_candidate_set_weight"]) != 0.0:
        raise RuntimeError("misaligned 256-unit candidate hard-set loss must remain disabled")
    if float(config["direct_predictor_candidate_coverage_weight"]) <= 0.0:
        raise RuntimeError("joint candidate head requires a positive top192 coverage loss")
    if str(config.get("direct_candidate_head_supervision")) != JOINT_CANDIDATE_COVERAGE_SEMANTICS:
        raise RuntimeError("joint candidate-head physical coverage semantics changed")
    if int(config.get("expected_validation_unique_invocations", -1)) != 69:
        raise RuntimeError("locked validation scope must contain exactly 69 invocations")
    if int(config.get("expected_validation_layer_expert_cells", -1)) != 12:
        raise RuntimeError("locked validation scope must contain exactly 12 layer/expert cells")
    layers = tuple(map(int, config["layers"]))
    exact_mapping = config["locked_exact_sampled_experts"]
    if set(exact_mapping) != {str(layer) for layer in layers}:
        raise RuntimeError("locked exact expert mapping does not exactly cover configured layers")
    locked_cells = {
        (layer, int(expert))
        for layer in layers for expert in exact_mapping[str(layer)]
    }
    if len(locked_cells) != int(config["expected_validation_layer_expert_cells"]):
        raise RuntimeError("locked exact expert mapping has the wrong layer/expert cell count")
    expected_shared = int(config.get("expected_cross_capture_shared_request_ids", -1))
    if expected_shared < 0:
        raise RuntimeError("expected cross-capture shared request count must be nonnegative")
    locked_hashes = [
        str(config["locked_tree_sha256"]),
        *(str(value) for value in config["locked_capture_sha256"].values()),
        str(config["reference"]["config_sha256"]),
        str(config["reference"]["index_sha256"]),
    ]
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in locked_hashes
    ):
        raise RuntimeError("every locked scientific hash must be exactly 64 lowercase hex characters")


def _verify_request_metadata(data: Mapping[str, np.ndarray]) -> dict[str, int]:
    return base.verify_request_separation({
        "split": np.asarray(data["split"]),
        "request_id": np.asarray(data["request_id"]),
    })


def load_capture_request_metadata(path: Path) -> dict[str, np.ndarray]:
    """Load only split provenance; no scientific tensors or metrics are inspected."""
    with np.load(path, allow_pickle=False) as loaded:
        return {
            "split": np.asarray(loaded["split"]).copy(),
            "request_id": np.asarray(loaded["request_id"]).copy(),
        }


def _request_split_map(data: Mapping[str, np.ndarray]) -> dict[str, str]:
    _verify_request_metadata(data)
    split = np.asarray(data["split"]).astype(str)
    request = np.asarray(data["request_id"]).astype(str)
    if split.shape != request.shape:
        raise RuntimeError("request split metadata arrays must have equal shape")
    result: dict[str, str] = {}
    for request_id, label in zip(request, split):
        previous = result.setdefault(str(request_id), str(label))
        if previous != str(label):
            raise RuntimeError(f"request split leakage for {request_id}")
    return result


def audit_cross_capture_request_ids(
    exact: Mapping[str, np.ndarray],
    cross: Mapping[str, np.ndarray],
    *,
    expected_shared_request_ids: int | None = None,
) -> dict[str, Any]:
    """Fail on cross-source split disagreement while allowing same-split pairing."""
    exact_map = _request_split_map(exact)
    cross_map = _request_split_map(cross)
    shared = sorted(set(exact_map) & set(cross_map))
    incompatible = [
        {
            "request_id": request_id,
            "exact_checkpoint_split": exact_map[request_id],
            "cross_reference_split": cross_map[request_id],
        }
        for request_id in shared
        if exact_map[request_id] != cross_map[request_id]
    ]
    same_by_split = {
        label: [
            request_id for request_id in shared
            if exact_map[request_id] == label and cross_map[request_id] == label
        ]
        for label in ("train", "validation", "test")
    }
    train_to_nontrain = [
        record for record in incompatible
        if "train" in {
            record["exact_checkpoint_split"], record["cross_reference_split"],
        }
    ]
    expected_match = (
        True if expected_shared_request_ids is None
        else len(shared) == int(expected_shared_request_ids)
    )
    audit = {
        "exact_unique_request_ids": len(exact_map),
        "cross_unique_request_ids": len(cross_map),
        "shared_request_id_count": len(shared),
        "expected_shared_request_id_count": (
            None if expected_shared_request_ids is None
            else int(expected_shared_request_ids)
        ),
        "expected_shared_request_ids_match": expected_match,
        "shared_same_split_request_id_counts": {
            label: len(values) for label, values in same_by_split.items()
        },
        "shared_same_split_request_ids": same_by_split,
        "incompatible_overlap_count": len(incompatible),
        "incompatible_overlaps": incompatible,
        "cross_source_train_to_nontrain_overlap_count": len(train_to_nontrain),
        "cross_source_train_to_nontrain_overlaps": train_to_nontrain,
        "split_compatible": not incompatible,
    }
    if incompatible:
        raise RuntimeError(
            "cross-capture request split mismatch: "
            + json.dumps(incompatible, sort_keys=True)
        )
    if not expected_match:
        raise RuntimeError(
            f"cross-capture shared request count {len(shared)} != "
            f"expected {int(expected_shared_request_ids)}"
        )
    return audit


def require_locked_request_audit(
    fit_facts: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    audit = fit_facts.get("cross_capture_request_id_audit")
    if not isinstance(audit, dict) or audit != manifest.get("cross_capture_request_id_audit"):
        raise RuntimeError("fit facts/manifest cross-capture request audits are absent or differ")
    expected = int(config["expected_cross_capture_shared_request_ids"])
    if (
        audit.get("split_compatible") is not True
        or audit.get("expected_shared_request_ids_match") is not True
        or int(audit.get("shared_request_id_count", -1)) != expected
        or int(audit.get("incompatible_overlap_count", -1)) != 0
        or int(audit.get("cross_source_train_to_nontrain_overlap_count", -1)) != 0
        or audit.get("incompatible_overlaps") != []
        or audit.get("cross_source_train_to_nontrain_overlaps") != []
    ):
        raise RuntimeError("fit bundle does not prove cross-capture request split compatibility")
    return dict(audit)


def admit_capture_rows(
    data: Mapping[str, np.ndarray], allowed_splits: Iterable[str],
) -> dict[str, np.ndarray]:
    """Return a row-filtered copy and never admit test into a bounded pilot."""
    allowed = frozenset(map(str, allowed_splits))
    if not allowed or not allowed <= {"train", "validation"} or "test" in allowed:
        raise RuntimeError("bounded set-utility runner cannot admit test rows")
    _verify_request_metadata(data)
    split = np.asarray(data["split"]).astype(str)
    mask = np.isin(split, sorted(allowed))
    rows = len(split)
    result: dict[str, np.ndarray] = {}
    for name, raw in data.items():
        value = np.asarray(raw)
        result[name] = value[mask].copy() if value.ndim and value.shape[0] == rows else value.copy()
    observed = set(map(str, np.unique(result["split"])))
    if not observed <= allowed or "test" in observed:
        raise RuntimeError("capture row admission leaked a forbidden split")
    return result


def load_capture_admitted(path: Path, allowed_splits: Iterable[str]) -> dict[str, np.ndarray]:
    # The immutable NPZ is monolithic: materializing it necessarily reads
    # physical test rows before masking. Facts therefore claim only that test
    # rows are never admitted, used, or inspected for metrics -- never that
    # they were not physically loaded. Subsequent objects contain admitted
    # train/validation rows only.
    with np.load(path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    return admit_capture_rows(data, allowed_splits)


def layer_view(data: Mapping[str, np.ndarray], layer: int) -> dict[str, np.ndarray]:
    mask = np.asarray(data["layer"]) == int(layer)
    return {name: np.asarray(value)[mask] for name, value in data.items()}


def exact_experts(config: Mapping[str, Any], layer: int) -> dict[int, str]:
    return {
        int(expert): str(stratum)
        for expert, stratum in config["locked_exact_sampled_experts"][str(layer)].items()
    }


def all_fit_experts(config: Mapping[str, Any], layer: int) -> list[int]:
    exact = exact_experts(config, layer)
    cross = {
        int(expert)
        for expert in config["locked_cross_sampled_experts"][str(layer)]
    }
    return sorted(set(exact) | cross)


def decode_expert(
    checkpoint: Path, index: Mapping[str, str], trees: Mapping[str, Any], layer: int, expert: int,
) -> dict[str, list[np.ndarray]]:
    return prior.decode_expert(checkpoint, index, trees, layer, expert)


def load_fit_expert(
    checkpoint: Path, index: Mapping[str, str], trees: Mapping[str, Any], layer: int, expert: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    """Decode levels while retaining locked resident E8M0 block scales."""
    leaves = {
        projection: prior.load_compressed_mxfp4_expert(
            checkpoint, index, layer, expert, projection,
        )
        for projection in PROJECTIONS
    }
    decoded = {
        projection: [trees[projection].decode(leaves[projection], level) for level in (2, 3, 4)]
        for projection in PROJECTIONS
    }
    scales = {
        projection: np.asarray(leaves[projection].scales, np.float32)
        for projection in ("gate", "up")
    }
    return decoded, scales


def stable_rows(local: Mapping[str, np.ndarray], split: str, maximum: int) -> np.ndarray:
    """Round-robin rows over requests before taking second positions."""
    eligible = np.flatnonzero(np.asarray(local["split"]).astype(str) == str(split))
    if not len(eligible):
        return eligible
    request = np.asarray(local["request_id"]).astype(str)
    position = np.asarray(local.get("position", np.arange(len(request))), np.int64)
    ordered_requests = sorted(set(request[eligible].tolist()))
    groups = {
        key: sorted(
            (int(row) for row in eligible if request[row] == key),
            key=lambda row: (int(position[row]), row),
        )
        for key in ordered_requests
    }
    result: list[int] = []
    depth = 0
    limit = int(maximum)
    while True:
        added = False
        for key in ordered_requests:
            if depth < len(groups[key]):
                result.append(groups[key][depth])
                added = True
                if limit and len(result) == limit:
                    return np.asarray(result, np.int64)
        if not added:
            break
        depth += 1
    return np.asarray(result, np.int64)


def _teacher_example(
    matrices: Mapping[str, list[np.ndarray]], activation: np.ndarray,
    static_down_metric: np.ndarray, encoded_abc: Any, path_length: int,
) -> dict[str, np.ndarray]:
    """Build one teacher from expert-static metric/ABC metadata.

    The static metric and encoded ABC are deliberately precomputed once per
    (layer, expert, locked proxy), rather than rebuilt per activation.
    """
    q2 = tuple(matrices[name][0] for name in PROJECTIONS)
    q4 = tuple(matrices[name][2] for name in PROJECTIONS)
    outputs = factorized_unit_outputs(q2, q4, activation)
    gram = unit_correction_gram(outputs.h2, outputs.h4, static_down_metric)
    path = exact_full_target_fixed_greedy(gram, int(path_length))
    diagonal = np.maximum(np.diag(gram).astype(np.float64), 0.0)
    teacher_marginal = np.zeros(len(diagonal), dtype=np.float32)
    teacher_marginal[path.order] = np.asarray(path.marginal_gain, np.float32)
    path_utility = np.zeros(len(diagonal), dtype=np.float32)
    apply_prefix = min(192, int(path_length))
    path_utility[path.order[:apply_prefix]] = np.maximum(
        path.marginal_gain[:apply_prefix], 0.0,
    ).astype(np.float32)
    mask = path.mask(len(diagonal), apply_prefix)
    gate2 = np.asarray(q2[0] @ activation, np.float32)
    up2 = np.asarray(q2[1] @ activation, np.float32)
    runtime_activation = np.asarray(activation, np.float16).astype(np.float32)
    runtime_gate2 = np.asarray(q2[0] @ runtime_activation, np.float16)
    runtime_up2 = np.asarray(q2[1] @ runtime_activation, np.float16)
    runtime_h2 = np.asarray(
        silu(runtime_gate2.astype(np.float32)) * runtime_up2.astype(np.float32),
        np.float16,
    )
    return {
        "x": np.asarray(activation, np.float16),
        "x_teacher": np.asarray(activation, np.float32),
        "g2": runtime_gate2,
        "u2": runtime_up2,
        "h2": runtime_h2,
        "h4": np.asarray(outputs.h4, np.float16),
        "g2_teacher": gate2,
        "u2_teacher": up2,
        "h2_teacher": np.asarray(outputs.h2, np.float32),
        "h4_teacher": np.asarray(outputs.h4, np.float32),
        "abc": np.stack(
            (encoded_abc.a, encoded_abc.b, encoded_abc.c), axis=1,
        ).astype(np.float32),
        "diagonal_score": np.asarray(diagonal, np.float32),
        "teacher_marginal": teacher_marginal,
        "path_utility": path_utility,
        "path_order": np.asarray(path.order, np.uint16),
        "path_gain": np.asarray(path.marginal_gain, np.float32),
        "mask192_packed": np.packbits(mask, bitorder="little"),
        "base_damage": np.asarray(np.ones(len(diagonal)) @ gram @ np.ones(len(diagonal)), np.float64),
    }


def _teacher_pairings(
    exact_layer: Mapping[str, np.ndarray], cross_layer: Mapping[str, np.ndarray],
    expert: int, config: Mapping[str, Any],
) -> list[tuple[str, str, int]]:
    """Return (cohort, activation source, local row), preserving domain shift."""
    exact_limit = int(config.get(
        "max_exact_teacher_rows_per_layer", config["max_exact_train_rows_per_layer"],
    ))
    cross_limit = int(config.get(
        "max_cross_teacher_rows_per_layer", config["max_cross_train_rows_per_layer"],
    ))
    exact_rows = stable_rows(exact_layer, "train", exact_limit)
    cross_rows = stable_rows(cross_layer, "train", cross_limit)
    routed, _ = occurrence(exact_layer, int(expert), "train", int(config["max_routed_train_rows_per_expert"]))
    result = [
        (FIT_COHORTS[0], "exact_checkpoint", int(row)) for row in routed
    ]
    result.extend((FIT_COHORTS[1], "exact_checkpoint", int(row)) for row in exact_rows)
    result.extend((FIT_COHORTS[2], "exact_checkpoint", int(row)) for row in exact_rows)
    result.extend((FIT_COHORTS[2], "cross_reference", int(row)) for row in cross_rows)
    return result


def _stack_teacher_rows(rows: Sequence[Mapping[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    if not rows:
        return
    for key in (
        "x", "g2", "u2", "h2", "h4", "abc", "diagonal_score", "teacher_marginal", "path_utility",
        "path_order", "path_gain", "mask192_packed", "base_damage",
    ):
        arrays[f"teacher__{key}"] = np.stack([np.asarray(row[key]) for row in rows])
    for key in ("expert_id", "activation_row"):
        arrays[f"teacher__{key}"] = np.asarray([row[key] for row in rows], np.int32)
    for key in ("cohort", "activation_source", "request_id"):
        arrays[f"teacher__{key}"] = np.asarray([str(row[key]) for row in rows], dtype="U96")


def fit_template_arrays(
    teacher_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Fit separate per-(cohort, expert) banks; unit IDs never cross experts."""
    index: list[dict[str, Any]] = []
    requested_by_kind = {
        FIT_COHORTS[0]: list(map(int, config["template_counts_primary"])),
        FIT_COHORTS[1]: list(map(int, config["template_counts_augmented"])),
        FIT_COHORTS[2]: list(map(int, config["template_counts_augmented"])),
    }
    experts = sorted({int(row["expert_id"]) for row in teacher_rows})
    for cohort in FIT_COHORTS:
        for expert in experts:
            selected = [
                row for row in teacher_rows
                if str(row["cohort"]) == cohort and int(row["expert_id"]) == expert
            ]
            if not selected:
                continue
            masks = np.stack([
                np.unpackbits(np.asarray(row["mask192_packed"], np.uint8), bitorder="little")[:UNITS]
                for row in selected
            ]).astype(bool)
            utilities = np.stack([np.asarray(row["path_utility"], np.float64) for row in selected])
            maximum = min(max(requested_by_kind[cohort]), len(selected))
            bank = build_nested_template_bank(
                np.full(len(selected), expert, np.int32), masks, utilities, maximum,
            ).for_expert(expert)
            key = f"template__{cohort}__e{expert}"
            arrays[key] = np.packbits(bank.masks, axis=1, bitorder="little")
            index.append({
                "cohort": cohort,
                "expert_id": expert,
                "array_key": key,
                "training_examples": len(selected),
                "training_unique_requests": len({
                    f"{row['activation_source']}:{row['request_id']}" for row in selected
                }),
                "available_templates": len(bank.masks),
                "requested_template_counts": requested_by_kind[cohort],
                "pairing_semantics": (
                    "actual_routed_occurrences" if cohort == FIT_COHORTS[0]
                    else "synthetic_all_x_by_expert_augmentation"
                ),
            })
    return index


def template_cohort_example_counts_by_expert(
    teacher_rows: Sequence[Mapping[str, Any]], experts: Sequence[int],
) -> dict[str, dict[str, int]]:
    """Emit the complete expert x cohort grid, including legitimate zero counts."""
    expert_ids = tuple(sorted({int(expert) for expert in experts}))
    counts = {
        str(expert): {cohort: 0 for cohort in FIT_COHORTS}
        for expert in expert_ids
    }
    for row in teacher_rows:
        expert = int(row["expert_id"])
        cohort = str(row["cohort"])
        if str(expert) not in counts:
            raise ValueError("teacher row names an expert outside the locked exact set")
        if cohort not in FIT_COHORTS:
            raise ValueError("teacher row names an unknown fit cohort")
        counts[str(expert)][cohort] += 1
    return counts


def _pq_fit(
    matrices: Mapping[int, Mapping[str, list[np.ndarray]]],
    block_scales: Mapping[int, Mapping[str, np.ndarray]],
    train_x: np.ndarray,
    config: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
    device: torch.device | str = "cpu",
) -> list[dict[str, Any]]:
    """Fit distinct layer-shared gate and up banks in locked E8M0 code space."""
    entries: list[dict[str, Any]] = []
    diagnostic_count = min(len(train_x), 64)
    diagnostic_indices = np.linspace(
        0, len(train_x) - 1, diagnostic_count, dtype=np.int64,
    )
    diagnostic_x = torch.as_tensor(
        np.asarray(train_x[diagnostic_indices], np.float32),
        dtype=torch.float32, device=torch.device(device),
    )
    for stages in map(int, config["pq_additive_stages"]):
        fitted_by_projection = {}
        codebook_keys = {}
        shared_bytes = 0
        for projection in ("gate", "up"):
            residual = np.concatenate([
                np.asarray(
                    matrices[expert][projection][2] - matrices[expert][projection][0],
                    np.float32,
                )
                for expert in sorted(matrices)
            ], axis=0)
            pooled_scales = np.concatenate([
                np.asarray(block_scales[expert][projection], np.float32)
                for expert in sorted(matrices)
            ], axis=0)
            fitted = fit_activation_weighted_block_pq(
                residual, train_x, block_scales=pooled_scales, stages=stages,
                max_iterations=int(config["pq_max_iterations"]),
            )
            fitted_by_projection[projection] = fitted
            codebook_key = f"pq_s{stages}__{projection}__codebooks"
            codebook_keys[projection] = codebook_key
            arrays[codebook_key] = np.asarray(fitted.centroids, np.float16)
            shared_bytes += int(arrays[codebook_key].nbytes)
        for expert in sorted(matrices):
            record: dict[str, Any] = {
                "config_id": f"block_pq_s{stages}_separate_gate_up",
                "selector_family": "block_pq_residual_synopsis",
                "stages": stages,
                "expert_id": expert,
                "codebook_keys": codebook_keys,
                "codebook_sharing": "separate_gate_up_layer_shared",
                "layer_shared_bytes": shared_bytes,
                "projection_arrays": {},
                "block_scale_provenance": "locked_resident_e8m0",
                "pq_fit_objective": (
                    "decoded_E8M0_scale_squared_activation_second_moment_weighted_"
                    "per_block_distortion;cross_block_covariance_omitted"
                ),
                "pq_whole_response_diagnostic_rows": diagnostic_count,
                "pq_whole_response_diagnostic_sampling": (
                    "deterministic_evenly_spaced_train_rows_across_exact_plus_cross_matrix"
                ),
            }
            expert_bytes = 0
            resident_scale_bytes_read = 0
            response_numerator = response_denominator = 0.0
            for projection in ("gate", "up"):
                value = np.asarray(
                    matrices[expert][projection][2] - matrices[expert][projection][0],
                    np.float32,
                )
                encoded = encode_block_residual_pq(
                    value, fitted_by_projection[projection],
                    block_scales=np.asarray(block_scales[expert][projection], np.float32),
                    block_scale_provenance="locked_resident_e8m0",
                )
                key = f"pq_s{stages}__e{expert}__{projection}__packed"
                scale_key = f"pq_s{stages}__e{expert}__{projection}__resident_scale_codes"
                arrays[key] = encoded.packed_codes
                arrays[scale_key] = np.asarray(encoded.block_scale_codes, np.uint8)
                expert_bytes += encoded.encoded_index_bytes
                resident_scale_bytes_read += encoded.block_scale_bytes_read
                decoded_residual = decode_block_residual_pq(encoded)
                with torch.no_grad():
                    actual_weight = torch.as_tensor(
                        value, dtype=torch.float32, device=diagnostic_x.device,
                    )
                    decoded_weight = torch.as_tensor(
                        decoded_residual, dtype=torch.float32, device=diagnostic_x.device,
                    )
                    actual_response = diagnostic_x @ actual_weight.T
                    decoded_response = diagnostic_x @ decoded_weight.T
                    numerator = float(torch.square(actual_response - decoded_response).sum().item())
                    denominator = float(torch.square(actual_response).sum().item())
                record[f"training_whole_response_relative_mse_{projection}"] = (
                    numerator / max(denominator, 1e-30)
                )
                response_numerator += numerator
                response_denominator += denominator
                record["projection_arrays"][projection] = {
                    "packed_key": key,
                    "scale_key": scale_key,
                    "code_shape": list(encoded.code_shape),
                    "weight_shape": list(encoded.weight_shape),
                }
            record["expert_specific_bytes"] = expert_bytes
            record["new_scale_metadata_bytes"] = 0
            record["resident_scale_bytes_read"] = resident_scale_bytes_read
            record["training_whole_response_relative_mse"] = (
                response_numerator / max(response_denominator, 1e-30)
            )
            entries.append(record)
    return entries


def _high_rank_fit(
    matrices: Mapping[int, Mapping[str, list[np.ndarray]]], train_x: np.ndarray,
    config: Mapping[str, Any], device: torch.device, arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Fit a bounded high-rank/low-bit diagnostic curve.

    The analysis basis and ordinary least-squares synthesis are frozen first;
    every reported low-bit synthesis is post-fit quantization, never QAT.
    These sampled-expert fits are transductive diagnostics, not evidence of
    bank-wide generalization.
    """
    residuals = {
        expert: (
            np.asarray(value["gate"][2] - value["gate"][0], np.float32),
            np.asarray(value["up"][2] - value["up"][0], np.float32),
        ) for expert, value in matrices.items()
    }
    gate, up = prior.gpu_responses(train_x, residuals, device)
    requested = list(config["high_rank_low_bit_controls"])
    identifiable = [
        record for record in requested
        if int(record["rank"]) <= min(train_x.shape)
    ]
    if not identifiable:
        raise RuntimeError(f"no requested rank is identifiable from train matrix {train_x.shape}")
    maximum = max(int(record["rank"]) for record in identifiable)
    desired = prior.canonical_subspace([*gate.values(), *up.values()], maximum, device)
    analysis = prior.analysis_from_subspace(train_x, desired, float(config["ridge_relative"]))
    entries: list[dict[str, Any]] = []
    packed_encodings = {"int2", "ternary", "sign_scale"}
    inherited_encodings = {"fp8_e4m3fn_per_row", "int8_per_row"}
    for specification in requested:
        rank = int(specification["rank"])
        encoding = str(specification["encoding"])
        if rank > min(train_x.shape):
            # Rank-512 is deliberately skipped unless the frozen training
            # design identifies it; it is non-promotable even when present.
            continue
        if encoding not in packed_encodings | inherited_encodings:
            raise RuntimeError(f"unsupported bounded high-rank encoding: {encoding}")
        key = f"high_rank_r{rank}_{encoding}"
        b = np.asarray(analysis[:rank], np.float16)
        arrays[f"{key}__analysis"] = b
        latent = np.asarray(train_x, np.float32) @ b.astype(np.float32).T
        singular = np.linalg.svd(latent.astype(np.float64), compute_uv=False)
        minimum_singular = float(singular[-1]) if len(singular) else 0.0
        condition = float(singular[0] / max(minimum_singular, 1e-30)) if len(singular) else float("inf")
        fitted = prior.syntheses(latent, {
            **{f"gate_{expert}": gate[expert] for expert in sorted(gate)},
            **{f"up_{expert}": up[expert] for expert in sorted(up)},
        }, device)
        for expert in sorted(matrices):
            record: dict[str, Any] = {
                "config_id": key,
                "selector_family": "high_rank_low_bit_linear_response",
                "rank": rank,
                "encoding": encoding,
                "expert_id": expert,
                "analysis_key": f"{key}__analysis",
                "layer_shared_bytes": int(b.nbytes),
                "projection_arrays": {},
                "fit_objective": "ordinary_response_least_squares",
                "quantization_training": "post_fit_row_quantization_not_qat",
                "training_samples": int(len(train_x)),
                "analysis_design_rank_identifiable": True,
                "analysis_design_condition_number": condition,
                "analysis_design_min_singular_value": minimum_singular,
                "sampled_expert_scope": (
                    "transductive_sampled_expert_diagnostic_not_bankwide_generalization"
                ),
                "promotable": False,
                "control_role": str(specification.get("control_role", "bounded_diagnostic")),
            }
            expert_bytes = 0
            total_num = total_den = 0.0
            projection_mse: dict[str, float] = {}
            for projection, response_bank in (("gate", gate), ("up", up)):
                synthesis = np.asarray(fitted[f"{projection}_{expert}"], np.float32)
                array_record: dict[str, Any]
                if encoding in packed_encodings:
                    encoded = quantize_low_bit_rows(synthesis, encoding)
                    packed_key = f"{key}__e{expert}__{projection}__packed"
                    scale_key = f"{key}__e{expert}__{projection}__scales"
                    arrays[packed_key] = encoded.packed_codes
                    arrays[scale_key] = encoded.scales
                    decoded = decode_low_bit_rows(encoded)
                    expert_bytes += encoded.storage_bytes
                    array_record = {
                        "serialization": "packed_row_codes_plus_fp16_scales",
                        "packed_key": packed_key,
                        "scale_key": scale_key,
                        "shape": list(encoded.shape),
                    }
                else:
                    encoded = prior.encode_array(synthesis, encoding)
                    decoded_key = f"{key}__e{expert}__{projection}__decoded_reference"
                    arrays[decoded_key] = np.asarray(encoded.decoded, np.float32)
                    decoded = np.asarray(encoded.decoded, np.float32)
                    expert_bytes += int(encoded.storage_bytes)
                    array_record = {
                        "serialization": "logical_row_scaled_payload_decoded_reference",
                        "decoded_key": decoded_key,
                        "shape": list(decoded.shape),
                    }
                actual = np.asarray(response_bank[expert], np.float32)
                predicted = latent @ decoded.T
                numerator = float(np.square(actual - predicted).sum())
                denominator = float(np.square(actual).sum())
                projection_mse[projection] = numerator / max(denominator, 1e-30)
                total_num += numerator
                total_den += denominator
                record["projection_arrays"][projection] = array_record
            record["expert_specific_bytes"] = expert_bytes
            record["training_gate_response_relative_mse"] = projection_mse["gate"]
            record["training_up_response_relative_mse"] = projection_mse["up"]
            record["training_response_relative_mse"] = total_num / max(total_den, 1e-30)
            entries.append(record)
    return entries


def _standardize_feature(
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(value, np.float32)
    mean = data.mean(axis=(0, 1), keepdims=True)
    scale = data.std(axis=(0, 1), keepdims=True)
    scale = np.maximum(scale, np.float32(1e-6))
    return (data - mean) / scale, mean.reshape(-1), scale.reshape(-1)


def _fp16_round_widen_pq_response(value: np.ndarray) -> np.ndarray:
    """Materialize PQ response features at the declared two-byte precision."""
    raw = np.asarray(value, np.float32)
    if not np.all(np.isfinite(raw)):
        raise ValueError("PQ response contains a non-finite value before FP16 rounding")
    rounded = np.asarray(raw, np.float16)
    if not np.all(np.isfinite(rounded)):
        raise ValueError("PQ response overflows the declared FP16 runtime feature format")
    return rounded.astype(np.float32)


def _abc_feature_transform(value: np.ndarray) -> np.ndarray:
    """Stable algebraic ABC features fitted/used without validation statistics."""
    abc = np.asarray(value, np.float32)
    a = np.maximum(abc[..., 0], np.float32(1e-20))
    b = abc[..., 1]
    c = np.maximum(abc[..., 2], np.float32(1e-20))
    correlation = np.clip(b / np.sqrt(a * c), -4.0, 4.0)
    return np.stack((np.log(a), correlation, np.log(c)), axis=-1).astype(np.float32)


def _teacher_marginal_transform(value: np.ndarray) -> np.ndarray:
    """Per-example signed log1p marginal in positive-median units."""
    marginal = np.asarray(value, np.float32)
    result = np.empty_like(marginal)
    for row in range(len(marginal)):
        positive = marginal[row][marginal[row] > 0.0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        scale = max(scale, 1e-20)
        result[row] = np.sign(marginal[row]) * np.log1p(np.abs(marginal[row]) / scale)
    return result


def _stable_model_seed(base_seed: int, *identity: object) -> int:
    payload = ":".join((str(int(base_seed)), *(str(value) for value in identity)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little") % (2**31)


def _strict_primary_compute_gate_pass(total_macs: int, gate_macs: int) -> bool:
    return int(total_macs) < int(gate_macs)


def _direct_predictor_fit(
    layer: int,
    teacher_rows: Sequence[Mapping[str, Any]],
    static_gram: Mapping[int, np.ndarray],
    pq_entries: Sequence[Mapping[str, Any]],
    arrays: dict[str, np.ndarray],
    config: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Fit a fixed two-objective direct-selector grid with no validation stopping."""
    entries: list[dict[str, Any]] = []
    pq_by_expert = {
        int(entry["expert_id"]): entry
        for entry in pq_entries
        if int(entry["stages"]) == 1
    }
    # Bounded, predeclared grid: two objectives, one nested-head control,
    # and one explicitly transductive train-only static-unit-bias upper bound.
    variants = (
        ("q2_abc_control", False, False, "listwise_boundary", "single"),
        ("q2_abc_pq1", True, False, "listwise_boundary", "single"),
        ("q2_abc_pq1", True, False, "listwise_boundary_hard_set", "single"),
        ("q2_abc_pq1_joint_nested", True, False, "listwise_boundary_hard_set", "joint_nested"),
        ("q2_abc_static_unit_bias_control", False, True, "listwise_boundary_hard_set", "single"),
    )
    for cohort in map(str, config["direct_predictor_training_cohorts"]):
        selected = [row for row in teacher_rows if str(row["cohort"]) == cohort]
        if not selected:
            continue
        # Deployable selector inputs are physically FP16.  The reference
        # evaluates their rounded values in FP32, matching the 2-byte reads.
        q2_raw = np.stack([
            np.stack((
                np.asarray(row["g2"], np.float16).astype(np.float32),
                np.asarray(row["u2"], np.float16).astype(np.float32),
                np.asarray(row["h2"], np.float16).astype(np.float32),
            ), axis=1)
            for row in selected
        ])
        abc_base_raw = _abc_feature_transform(
            np.stack([np.asarray(row["abc"], np.float32) for row in selected]),
        )
        q2, q2_mean, q2_scale = _standardize_feature(q2_raw)
        bias_by_expert: dict[int, np.ndarray] = {}
        bias_keys: dict[str, str] = {}
        for expert in sorted({int(row["expert_id"]) for row in selected}):
            expert_rows = [row for row in selected if int(row["expert_id"]) == expert]
            supports = np.stack([
                np.unpackbits(
                    np.asarray(row["mask192_packed"], np.uint8), bitorder="little",
                )[:UNITS]
                for row in expert_rows
            ])
            frequency = np.asarray(supports.mean(axis=0), np.float16)
            bias_by_expert[expert] = frequency.astype(np.float32)
            bias_key = f"direct__{cohort}__e{expert}__train_selection_frequency_fp16"
            arrays[bias_key] = frequency
            bias_keys[str(expert)] = bias_key
        static_bias_raw = np.stack([
            bias_by_expert[int(row["expert_id"])] for row in selected
        ])[..., None]
        gram = np.stack([
            unit_correction_gram(
                np.asarray(row["h2_teacher"], np.float32),
                np.asarray(row["h4_teacher"], np.float32),
                static_gram[int(row["expert_id"])],
            )
            for row in selected
        ]).astype(np.float32)
        teacher_marginal = _teacher_marginal_transform(np.stack([
            np.asarray(row["teacher_marginal"], np.float32) for row in selected
        ]))
        teacher_order = np.stack([
            np.asarray(row["path_order"], np.int64) for row in selected
        ])
        request_ids = tuple(
            f"{row['activation_source']}:{row['request_id']}" for row in selected
        )
        split_labels = tuple("train" for _ in selected)
        pq_raw = _fp16_round_widen_pq_response(
            np.stack([
                np.stack(_selector_prediction(
                    arrays,
                    pq_by_expert[int(row["expert_id"])],
                    np.asarray(row["x"], np.float16).astype(np.float32),
                )[:2], axis=1)
                for row in selected
            ])
        )
        pq, pq_mean, pq_scale = _standardize_feature(pq_raw)
        for feature_name, use_pq, use_static_bias, objective, head_mode in variants:
            key = f"direct_{cohort}_{feature_name}_{objective}_{head_mode}"
            abc_variant_raw = (
                np.concatenate((abc_base_raw, static_bias_raw), axis=-1)
                if use_static_bias else abc_base_raw
            )
            abc, abc_mean, abc_scale = _standardize_feature(abc_variant_raw)
            abc_dim = int(abc.shape[-1])
            model_seed = _stable_model_seed(
                int(config["seed"]), int(layer), cohort, feature_name, objective, head_mode,
            )
            # The training helper seeds its optimization loop, but model
            # parameters are initialized before that helper is called.
            torch.manual_seed(model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model_seed)
            model = CompactCoherentUnitPredictor(
                3, pq_delta_dim=2 if use_pq else 0, abc_dim=abc_dim,
                hidden_dim=int(config["direct_predictor_hidden_dim"]),
                unit_count=UNITS, head_mode=head_mode,
            ).to(device)
            training = DirectSetTrainingData(
                q2_unit_features=torch.as_tensor(q2, device=device),
                teacher_marginal=torch.as_tensor(teacher_marginal, device=device),
                teacher_gram=torch.as_tensor(gram, device=device),
                teacher_order=torch.as_tensor(teacher_order, device=device),
                request_ids=request_ids,
                split_labels=split_labels,
                pq_delta_response=torch.as_tensor(pq, device=device) if use_pq else None,
                abc=torch.as_tensor(abc, device=device),
            )
            set_enabled = objective == "listwise_boundary_hard_set"
            training_config = DirectSetTrainingConfig(
                epochs=int(config["direct_predictor_epochs"]),
                learning_rate=float(config["direct_predictor_learning_rate"]),
                weight_decay=float(config["direct_predictor_weight_decay"]),
                apply_count=int(config["applied_units"]),
                candidate_count=int(config["candidate_units"]),
                boundary_width=int(config["direct_predictor_boundary_width"]),
                temperature=float(config["direct_predictor_temperature"]),
                marginal_weight=0.0,
                listwise_weight=1.0,
                boundary_weight=1.0,
                set_weight=float(config["direct_predictor_set_weight"]) if set_enabled else 0.0,
                candidate_set_weight=0.0,
                candidate_coverage_weight=(
                    float(config["direct_predictor_candidate_coverage_weight"])
                    if head_mode == "joint_nested" else 0.0
                ),
                seed=model_seed,
            )
            fitted = train_direct_set_predictor(model, training, training_config)
            state_keys: dict[str, str] = {}
            for name, tensor in model.state_dict().items():
                array_key = f"{key}__state__{name.replace('.', '__')}"
                arrays[array_key] = tensor.detach().cpu().numpy().astype(np.float16)
                state_keys[name] = array_key
            normalization = {
                "q2_mean": f"{key}__q2_mean", "q2_scale": f"{key}__q2_scale",
                "abc_mean": f"{key}__abc_mean", "abc_scale": f"{key}__abc_scale",
            }
            arrays[normalization["q2_mean"]] = q2_mean.astype(np.float16)
            arrays[normalization["q2_scale"]] = q2_scale.astype(np.float16)
            arrays[normalization["abc_mean"]] = abc_mean.astype(np.float16)
            arrays[normalization["abc_scale"]] = abc_scale.astype(np.float16)
            if use_pq:
                normalization.update({
                    "pq_mean": f"{key}__pq_mean", "pq_scale": f"{key}__pq_scale",
                })
                arrays[normalization["pq_mean"]] = pq_mean.astype(np.float16)
                arrays[normalization["pq_scale"]] = pq_scale.astype(np.float16)
            model_accounting = model.accounting(parameter_bits=16, abc_bits=16)
            pq_feature_compute_macs = 0
            if use_pq:
                _, _, pq_fit_compute = _selector_prediction(
                    arrays, pq_by_expert[int(selected[0]["expert_id"])],
                    np.zeros(INPUTS, np.float32),
                )
                pq_feature_compute_macs = int(pq_fit_compute["compute_macs"])
            direct_selector_compute_macs = int(
                model_accounting.linear_macs_per_invocation + pq_feature_compute_macs
            )
            independent_rerank_compute_macs = int(
                int(config["candidate_units"]) * (2 * INPUTS + 6)
            )
            total_primary_compute_macs = (
                direct_selector_compute_macs + independent_rerank_compute_macs
            )
            normalization_bytes = sum(
                int(np.asarray(arrays[value]).size * 2)
                for value in normalization.values()
            )
            pq_reference = (
                pq_by_expert if use_pq else {}
            )
            entries.append({
                "config_id": key,
                "selector_family": "direct_set_predictor",
                "training_cohort": cohort,
                "feature_variant": feature_name,
                "objective_variant": objective,
                "set_loss_enabled": set_enabled,
                "candidate_set_loss_enabled": False,
                "candidate_coverage_weight": (
                    float(config["direct_predictor_candidate_coverage_weight"])
                    if head_mode == "joint_nested" else 0.0
                ),
                "candidate_coverage_semantics": (
                    JOINT_CANDIDATE_COVERAGE_SEMANTICS
                    if head_mode == "joint_nested" else "disabled_single_head_control"
                ),
                "hidden_dim": int(config["direct_predictor_hidden_dim"]),
                "q2_unit_dim": 3,
                "pq_delta_dim": 2 if use_pq else 0,
                "abc_dim": abc_dim,
                "head_mode": head_mode,
                "unit_bias_keys": bias_keys if use_static_bias else {},
                "unit_bias_bytes_per_expert": UNITS * 2 if use_static_bias else 0,
                "static_unit_bias_semantics": (
                    "train_only_per_expert_unit_top192_selection_frequency_fp16;"
                    "transductive_sampled_expert_upper_bound"
                    if use_static_bias else "none"
                ),
                "state_keys": state_keys,
                "normalization": normalization,
                "experts": sorted({int(row["expert_id"]) for row in selected}),
                "pq_config_by_expert": {
                    str(expert): str(value["config_id"])
                    for expert, value in pq_reference.items()
                },
                "model_parameter_bytes_fp16": int(model_accounting.parameter_metadata_bytes),
                "model_serialization": "post_fit_fp16_rounding_evaluated_on_validation",
                "normalization_bytes_fp16": normalization_bytes,
                "layer_shared_bytes": int(
                    model_accounting.parameter_metadata_bytes + normalization_bytes
                ),
                "expert_specific_bytes": 0,
                "abc_bytes": int(model_accounting.static_abc_metadata_bytes + 3 * 4),
                "linear_macs": int(model_accounting.linear_macs_per_invocation),
                "pq_feature_compute_macs": pq_feature_compute_macs,
                "direct_selector_compute_macs": direct_selector_compute_macs,
                "independent_abc_rerank_compute_macs": independent_rerank_compute_macs,
                "total_primary_selector_rerank_compute_macs": total_primary_compute_macs,
                "primary_compute_gate_macs": int(
                    config["promotion_gates"]["max_selector_compute_macs"]
                ),
                "primary_compute_gate_pass": _strict_primary_compute_gate_pass(
                    total_primary_compute_macs, config["promotion_gates"]["max_selector_compute_macs"]
                ),
                "context_additions": int(model_accounting.context_reduction_additions),
                "training_examples": len(selected),
                "training_unique_requests": len(set(request_ids)),
                "feature_transform": (
                    "q2_and_pq_train_mean_std; "
                    "abc_logA_signedB_over_sqrtAC_logC"
                    + ("_plus_train_selection_frequency" if use_static_bias else "")
                    + "_then_train_mean_std"
                ),
                "teacher_marginal_transform": "per_example_signed_log1p_in_positive_median_units",
                "model_initialization_and_training_seed": model_seed,
                "seed_derivation": "sha256(base_seed,layer,cohort,feature_variant,objective_variant,head_mode)",
                "runtime_feature_precision": (
                    "Q2_and_PQ_FP16-rounded_values_evaluated_by_FP32_reference_kernels"
                    if use_pq else "Q2_FP16-rounded_values_evaluated_by_FP32_reference_kernels"
                ),
                "pq_runtime_response_precision": (
                    "fp16_round_then_fp32_reference" if use_pq else "not_applicable"
                ),
                "teacher_gram_precision": "FP32 h2_teacher/h4_teacher with transient FP64 Gram algebra",
                "epochs_fixed": int(config["direct_predictor_epochs"]),
                "validation_early_stopping": False,
                "initial_train_loss": float(fitted.initial_train_loss),
                "final_train_loss": float(fitted.final_train_loss),
                "training_history": [
                    {
                        "epoch": int(epoch.epoch),
                        "train_loss": float(epoch.train_loss),
                        "validation_loss": epoch.validation_loss,
                    }
                    for epoch in fitted.history
                ],
            })
            del model, training
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del gram
    return entries


def _fit_layer(
    layer: int, exact: Mapping[str, np.ndarray], cross: Mapping[str, np.ndarray],
    checkpoint: Path, index: Mapping[str, str], trees: Mapping[str, Any],
    config: Mapping[str, Any], device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    exact_layer, cross_layer = layer_view(exact, layer), layer_view(cross, layer)
    proxy, proxy_facts = base.proxy_gradients(
        exact_layer["xplus"], [exact_layer[f"h{h}_router_logits"] for h in range(1, 5)],
        exact_layer["split"],
    )
    fit_experts = {
        expert: load_fit_expert(checkpoint, index, trees, layer, expert)
        for expert in all_fit_experts(config, layer)
    }
    matrices = {expert: value[0] for expert, value in fit_experts.items()}
    block_scales = {expert: value[1] for expert, value in fit_experts.items()}
    rows: list[dict[str, Any]] = []
    exact_only = set(exact_experts(config, layer))
    beta = float(proxy_facts["beta"])
    # These terms depend only on the decoded expert and frozen proxy.  Their
    # O(unit^2 * output) construction is paid once, never per activation.
    static_down_metrics = {
        expert: down_metric_gram(
            matrices[expert]["down"][0], matrices[expert]["down"][2],
            proxy=proxy, beta=beta,
        )
        for expert in sorted(exact_only)
    }
    encoded_abc_by_expert: dict[int, Any] = {}
    abc_bytes_by_expert: dict[int, int] = {}
    for expert in sorted(exact_only):
        exact_abc = unit_score_metadata(
            matrices[expert]["down"][0], matrices[expert]["down"][2],
            proxy=proxy, beta=beta,
        )
        encoded_abc_by_expert[expert], abc_bytes_by_expert[expert] = (
            encode_unit_score_metadata(exact_abc, "fp16")
        )
    for expert in sorted(exact_only):
        for cohort, source, record in _teacher_pairings(exact_layer, cross_layer, expert, config):
            local = exact_layer if source == "exact_checkpoint" else cross_layer
            teacher = _teacher_example(
                matrices[expert], np.asarray(local["x"][record], np.float32),
                static_down_metrics[expert], encoded_abc_by_expert[expert],
                int(config["teacher_path_length"]),
            )
            rows.append({
                **teacher, "cohort": cohort, "activation_source": source,
                "activation_row": int(record), "expert_id": expert,
                "request_id": str(local["request_id"][record]),
            })
    arrays: dict[str, np.ndarray] = {}
    _stack_teacher_rows(rows, arrays)
    template_index = fit_template_arrays(rows, config, arrays)
    exact_train_rows = stable_rows(
        exact_layer, "train", int(config["max_exact_train_rows_per_layer"]),
    )
    cross_train_rows = stable_rows(
        cross_layer, "train", int(config["max_cross_train_rows_per_layer"]),
    )
    exact_train = np.asarray(exact_layer["x"][exact_train_rows], np.float16).astype(np.float32)
    cross_train = np.asarray(cross_layer["x"][cross_train_rows], np.float16).astype(np.float32)
    train_x = np.concatenate((exact_train, cross_train))
    pq_index = _pq_fit(
        matrices, block_scales, train_x, config, arrays, device=device,
    )
    high_rank_index = _high_rank_fit(matrices, train_x, config, device, arrays)
    direct_index = _direct_predictor_fit(
        layer, rows, static_down_metrics, pq_index, arrays, config, device,
    )
    summary = {
        "layer": layer,
        "teacher_rows": len(rows),
        "teacher_rows_by_cohort": {
            cohort: sum(str(row["cohort"]) == cohort for row in rows) for cohort in FIT_COHORTS
        },
        "teacher_unique_requests_by_cohort": {
            cohort: len({
                f"{row['activation_source']}:{row['request_id']}"
                for row in rows if str(row["cohort"]) == cohort
            })
            for cohort in FIT_COHORTS
        },
        "template_cohort_example_counts_by_expert": (
            template_cohort_example_counts_by_expert(rows, sorted(exact_only))
        ),
        "compact_teacher_bytes": int(sum(
            value.nbytes for key, value in arrays.items() if key.startswith("teacher__")
        )),
        "full_correction_tensors_serialized": False,
        "proxy_facts": proxy_facts,
        "expert_static_precompute": {
            "down_metric_gram_once_per_exact_expert": True,
            "encoded_fp16_abc_once_per_exact_expert": True,
            "abc_bytes_by_expert": {
                str(expert): int(value) for expert, value in abc_bytes_by_expert.items()
            },
        },
        "selector_runtime_activation_precision": "FP16-rounded values evaluated in FP32 reference kernels",
        "template_entries": template_index,
        "selector_entries": pq_index + high_rank_index + direct_index,
        "training_activation_rows": int(len(train_x)),
        "training_unique_requests": {
            "exact_checkpoint": int(len(set(map(str, exact_layer["request_id"][exact_train_rows])))),
            "cross_reference": int(len(set(map(str, cross_layer["request_id"][cross_train_rows])))),
        },
        "training_sources": ["exact_checkpoint", "cross_reference"],
        "fit_split": "train",
        "validation_or_test_rows_used": False,
    }
    return arrays, summary


def _locked_facts(
    config: Mapping[str, Any], config_path: Path, exact_hashes: Mapping[str, str],
    cross_hashes: Mapping[str, str], runner: Path,
) -> dict[str, Any]:
    return {
        "run_id": str(config["run_id"]),
        "config_sha256": sha256(config_path),
        "exact_capture_sha256": str(exact_hashes["capture_sha256"]),
        "cross_capture_sha256": str(cross_hashes["capture_sha256"]),
        "tree_sha256": str(exact_hashes["tree_sha256"]),
        "checkpoint_config_sha256": str(exact_hashes["config_sha256"]),
        "checkpoint_index_sha256": str(exact_hashes["index_sha256"]),
        "runner_sha256": sha256(runner),
        "teacher_core_sha256": sha256(EXPERIMENT / "src/oracle_study/unit_set_teacher.py"),
        "oracle_core_sha256": sha256(EXPERIMENT / "src/oracle_study/set_utility_oracles.py"),
        "selector_core_sha256": sha256(EXPERIMENT / "src/oracle_study/set_utility_selector.py"),
        "direct_predictor_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/direct_set_predictor.py"
        ),
        "neuron_selector_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/neuron_selector.py"
        ),
        "prior_runner_sha256": sha256(
            EXPERIMENT / "scripts/run_neuron_selector_distillation.py"
        ),
        "base_runner_sha256": sha256(
            EXPERIMENT / "scripts/run_sparse_streaming_study.py"
        ),
        "sparse_streaming_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/sparse_streaming.py"
        ),
        "mxfp4_embed_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/mxfp4_embed.py"
        ),
        "selective_pages_runner_sha256": sha256(
            EXPERIMENT / "scripts/run_mxfp4_selective_pages.py"
        ),
        "phase_a_runner_sha256": sha256(
            EXPERIMENT / "scripts/run_phase_a_remote.py"
        ),
    }


def fit_phase(args: argparse.Namespace, config: dict[str, Any]) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    audit_exact, hashes_exact = base.verify_locked_inputs(
        config, args.config, args.exact_captures, args.checkpoint, args.trees, "exact_checkpoint",
    )
    audit_cross, hashes_cross = base.verify_locked_inputs(
        config, args.config, args.cross_captures, args.checkpoint, args.trees, "cross_reference",
    )
    request_audit = audit_cross_capture_request_ids(
        load_capture_request_metadata(args.exact_captures),
        load_capture_request_metadata(args.cross_captures),
        expected_shared_request_ids=int(config["expected_cross_capture_shared_request_ids"]),
    )
    exact = load_capture_admitted(args.exact_captures, ["train"])
    cross = load_capture_admitted(args.cross_captures, ["train"])
    device = torch.device(args.device)
    hardware = runtime_provenance(device)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {projection: tree_from_record(tree_records[projection]) for projection in PROJECTIONS}
    locked = _locked_facts(config, args.config, hashes_exact, hashes_cross, Path(__file__).resolve())
    facts_path = args.output / "fit_facts.json"
    completed: list[int] = []
    failures: list[dict[str, Any]] = []
    if facts_path.exists():
        old = json.loads(facts_path.read_text())
        for key, value in locked.items():
            if old.get(key) != value:
                raise RuntimeError(f"fit resume provenance mismatch for {key}")
        if old.get("cross_capture_request_id_audit") != request_audit:
            raise RuntimeError("fit resume cross-capture request audit mismatch")
        require_runtime_provenance_match(
            old.get("runtime_provenance"), hardware, "fit resume",
        )
        if old.get("completed") is True:
            raise RuntimeError("fit output is already complete")
        completed = list(map(int, old.get("completed_layers", [])))
        failures = list(old.get("failure_history", [])) + list(old.get("failures", []))
    facts = {
        **locked, "phase": "fit", "completed": False, "completed_layers": completed,
        "failures": [], "failure_history": failures,
        "fit_split": "train", "validation_or_test_rows_used": False,
        "test_rows_admitted": False, "test_rows_used": False,
        "monolithic_source_contains_test_rows": True,
        "request_separation_verified": True,
        "cross_capture_request_id_audit": request_audit,
        "direct_candidate_head_supervision": JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "runtime_provenance": hardware,
        "codec_locked": True, "checkpoint_audit_exact": audit_exact,
        "checkpoint_audit_cross": audit_cross, "device": hardware["device"],
        "gpu": hardware["gpu_names_used"][0] if hardware["gpu_names_used"] else None,
        "host": hardware["host"],
    }
    atomic_json(facts_path, facts)
    try:
        for layer in map(int, config["layers"]):
            if layer in completed:
                continue
            arrays, summary = _fit_layer(
                layer, exact, cross, args.checkpoint, index, trees, config, device,
            )
            shard = args.output / f"set_utility_fit_layer_{layer}.npz"
            sidecar = args.output / f"set_utility_fit_layer_{layer}.json"
            atomic_npz(shard, arrays)
            summary.update({
                "shard": shard.name, "shard_sha256": sha256(shard),
                "shard_bytes": shard.stat().st_size, "array_count": len(arrays),
                "runtime_provenance": hardware,
            })
            atomic_json(sidecar, summary)
            completed.append(layer)
            facts["completed_layers"] = completed
            atomic_json(facts_path, facts)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        layers = {}
        for layer in map(int, config["layers"]):
            sidecar_path = args.output / f"set_utility_fit_layer_{layer}.json"
            sidecar = json.loads(sidecar_path.read_text())
            require_runtime_provenance_match(
                sidecar.get("runtime_provenance"), hardware, f"fit layer {layer} sidecar",
            )
            shard = args.output / str(sidecar["shard"])
            if sha256(shard) != sidecar["shard_sha256"]:
                raise RuntimeError(f"fit shard {layer} changed before manifest finalization")
            layers[str(layer)] = {**sidecar, "sidecar_sha256": sha256(sidecar_path)}
        manifest_path = args.output / "set_utility_fit_manifest.json"
        atomic_json(manifest_path, {
            "schema_version": 1, "run_id": config["run_id"],
            "config_sha256": sha256(args.config), "fit_split": "train",
            "validation_or_test_rows_used": False, "test_rows_admitted": False,
            "test_rows_used": False, "monolithic_source_contains_test_rows": True,
            "cross_capture_request_id_audit": request_audit,
            "direct_candidate_head_supervision": JOINT_CANDIDATE_COVERAGE_SEMANTICS,
            "runtime_provenance": hardware,
            "layers": layers,
        })
        facts["manifest_sha256"] = sha256(manifest_path)
        facts["completed"] = True
        facts["completed_unix"] = time.time()
        atomic_json(facts_path, facts)
    except Exception as error:
        facts["failures"].append({
            "type": type(error).__name__, "message": str(error), "unix": time.time(),
        })
        atomic_json(facts_path, facts)
        raise


def load_fit_layer(
    fit_dir: Path, manifest: Mapping[str, Any], layer: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    record = manifest["layers"][str(int(layer))]
    name = Path(str(record["shard"]))
    if name.is_absolute() or name.name != str(name):
        raise RuntimeError("fit shard path is not a local basename")
    path = fit_dir / name
    if sha256(path) != str(record["shard_sha256"]):
        raise RuntimeError(f"fit shard {layer} changed")
    if path.stat().st_size != int(record["shard_bytes"]):
        raise RuntimeError(f"fit shard {layer} byte count changed")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if len(arrays) != int(record["array_count"]):
        raise RuntimeError(f"fit shard {layer} array count changed")
    return arrays, dict(record)


def evaluation_plan(
    data: Mapping[str, np.ndarray], config: Mapping[str, Any],
) -> list[tuple[int, int, str, np.ndarray, np.ndarray]]:
    if set(map(str, np.unique(data["split"]))) - {"train", "validation"}:
        raise RuntimeError("evaluation data admitted a forbidden split")
    result = []
    maximum = int(config["max_validation_invocations_per_expert"])
    for layer in map(int, config["layers"]):
        local = layer_view(data, layer)
        for expert, stratum in sorted(exact_experts(config, layer).items()):
            records, ranks = occurrence(local, expert, "validation", maximum)
            if len(records):
                result.append((layer, expert, stratum, records, ranks))
    return result


def locked_validation_cells(config: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(
        (int(layer), int(expert))
        for layer, experts in config["locked_exact_sampled_experts"].items()
        for expert in experts
    ))


def audit_validation_plan(
    plan: Sequence[tuple[int, int, str, np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze broad validation coverage before any scientific evaluation runs."""
    expected_cells = locked_validation_cells(config)
    observed_cells = tuple(sorted((int(layer), int(expert)) for layer, expert, *_ in plan))
    observed_invocations = sum(len(records) for _, _, _, records, _ in plan)
    expected_invocations = int(config["expected_validation_unique_invocations"])
    expected_cell_count = int(config["expected_validation_layer_expert_cells"])
    if len(expected_cells) != expected_cell_count:
        raise RuntimeError(
            f"locked validation cells {len(expected_cells)} != expected {expected_cell_count}"
        )
    if len(set(observed_cells)) != len(observed_cells):
        raise RuntimeError("validation plan contains duplicate layer/expert cells")
    if observed_cells != expected_cells:
        raise RuntimeError(
            f"observed validation cells {observed_cells!r} != locked {expected_cells!r}"
        )
    if observed_invocations != expected_invocations:
        raise RuntimeError(
            f"observed validation invocations {observed_invocations} "
            f"!= expected {expected_invocations}"
        )
    return {
        "expected_unique_invocations": expected_invocations,
        "observed_unique_invocations": observed_invocations,
        "expected_layer_expert_cells": expected_cell_count,
        "observed_layer_expert_cells": len(observed_cells),
        "locked_layer_expert_cells": [list(cell) for cell in expected_cells],
        "observed_layer_expert_cell_records": [
            {
                "layer": int(layer),
                "expert_id": int(expert),
                "invocations": len(records),
            }
            for layer, expert, _, records, _ in plan
        ],
        "validation_scope_complete": True,
    }


def _template_masks(arrays: Mapping[str, np.ndarray], entry: Mapping[str, Any], count: int) -> np.ndarray:
    packed = np.asarray(arrays[str(entry["array_key"])], np.uint8)
    masks = np.unpackbits(packed, axis=1, bitorder="little")[:, :UNITS].astype(bool)
    return masks[: min(int(count), len(masks))]


def _pq_encoded(
    arrays: Mapping[str, np.ndarray], entry: Mapping[str, Any], projection: str,
) -> EncodedBlockResidualPQ:
    record = entry["projection_arrays"][projection]
    return EncodedBlockResidualPQ(
        stored_codebooks=np.asarray(
            arrays[str(entry["codebook_keys"][projection])], np.float16,
        ),
        packed_codes=np.asarray(arrays[str(record["packed_key"])], np.uint8),
        code_shape=tuple(map(int, record["code_shape"])),
        weight_shape=tuple(map(int, record["weight_shape"])),
        block_scale_codes=np.asarray(arrays[str(record["scale_key"])], np.uint8),
        block_scale_provenance="locked_resident_e8m0",
    )


def _high_rank_matrix(
    arrays: Mapping[str, np.ndarray], entry: Mapping[str, Any], projection: str,
) -> np.ndarray:
    record = entry["projection_arrays"][projection]
    if "decoded_key" in record:
        return np.asarray(arrays[str(record["decoded_key"])], np.float32)
    encoded = QuantizedRows(
        packed_codes=np.asarray(arrays[str(record["packed_key"])], np.uint8),
        scales=np.asarray(arrays[str(record["scale_key"])], np.float16),
        shape=tuple(map(int, record["shape"])),
        encoding=str(entry["encoding"]),
    )
    return decode_low_bit_rows(encoded)


def _direct_model(
    arrays: Mapping[str, np.ndarray], entry: Mapping[str, Any],
) -> CompactCoherentUnitPredictor:
    model = CompactCoherentUnitPredictor(
        int(entry["q2_unit_dim"]),
        pq_delta_dim=int(entry["pq_delta_dim"]),
        abc_dim=int(entry["abc_dim"]),
        hidden_dim=int(entry["hidden_dim"]),
        unit_count=UNITS,
        head_mode=str(entry["head_mode"]),
    )
    state = {
        name: torch.as_tensor(np.asarray(arrays[str(key)], np.float32))
        for name, key in entry["state_keys"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _normalization(
    arrays: Mapping[str, np.ndarray], entry: Mapping[str, Any], prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    record = entry["normalization"]
    return (
        np.asarray(arrays[str(record[f"{prefix}_mean"])], np.float32),
        np.asarray(arrays[str(record[f"{prefix}_scale"])], np.float32),
    )


def _direct_prediction(
    arrays: Mapping[str, np.ndarray],
    entry: Mapping[str, Any],
    layer_record: Mapping[str, Any],
    expert: int,
    activation: np.ndarray,
    q2_features: np.ndarray,
    abc_features: np.ndarray,
) -> tuple[PredictorOutput, dict[str, int]]:
    model = _direct_model(arrays, entry)
    q2_mean, q2_scale = _normalization(arrays, entry, "q2")
    abc_mean, abc_scale = _normalization(arrays, entry, "abc")
    q2_runtime = np.asarray(q2_features, np.float16).astype(np.float32)
    q2 = (q2_runtime - q2_mean) / q2_scale
    abc_runtime = _abc_feature_transform(np.asarray(abc_features, np.float32))
    unit_bias_keys = dict(entry.get("unit_bias_keys", {}))
    if unit_bias_keys:
        bias_key = unit_bias_keys.get(str(int(expert)))
        if bias_key is None:
            raise RuntimeError("transductive static-unit-bias model has no fitted expert bias")
        unit_bias = np.asarray(arrays[str(bias_key)], np.float16).astype(np.float32)[:, None]
        abc_runtime = np.concatenate((abc_runtime, unit_bias), axis=1)
    abc = (abc_runtime - abc_mean) / abc_scale
    pq_tensor = None
    pq_compute = {"compute_macs": 0, "compute_additions": 0, "scale_multiplications": 0, "bytes_read": 0}
    pq_expert_bytes = pq_shared_bytes = 0
    if int(entry["pq_delta_dim"]):
        candidates = [
            value for value in layer_record["selector_entries"]
            if str(value["selector_family"]) == "block_pq_residual_synopsis"
            and int(value["expert_id"]) == int(expert)
            and str(value["config_id"]) == str(entry["pq_config_by_expert"][str(int(expert))])
        ]
        if len(candidates) != 1:
            raise RuntimeError("direct predictor PQ feature has no unique fitted synopsis")
        pq_entry = candidates[0]
        runtime_activation = np.asarray(activation, np.float16).astype(np.float32)
        gate, up, pq_compute = _selector_prediction(arrays, pq_entry, runtime_activation)
        pq_value = _fp16_round_widen_pq_response(np.stack((gate, up), axis=1))
        pq_mean, pq_scale = _normalization(arrays, entry, "pq")
        pq_tensor = torch.as_tensor(((pq_value - pq_mean) / pq_scale)[None], dtype=torch.float32)
        pq_expert_bytes = int(pq_entry["expert_specific_bytes"])
        pq_shared_bytes = int(pq_entry["layer_shared_bytes"])
    with torch.no_grad():
        output = model(
            torch.as_tensor(q2[None], dtype=torch.float32),
            pq_delta_response=pq_tensor,
            abc=torch.as_tensor(abc[None], dtype=torch.float32),
        )
    compute = {
        "compute_macs": int(entry["linear_macs"]) + int(pq_compute["compute_macs"]),
        "compute_additions": int(entry["context_additions"]) + int(pq_compute["compute_additions"]),
        "scale_multiplications": int(pq_compute.get("scale_multiplications", 0)),
        "bytes_read": (
            int(entry["model_parameter_bytes_fp16"])
            + int(entry["normalization_bytes_fp16"])
            + int(pq_compute["bytes_read"])
            + FP16_BYTES * UNITS * (
                int(entry["q2_unit_dim"]) + int(entry["pq_delta_dim"])
            )
            + int(entry["abc_bytes"])
        ),
        "expert_metadata_bytes": pq_expert_bytes,
        "layer_shared_bytes": int(entry["layer_shared_bytes"]) + pq_shared_bytes,
        "abc_bytes": int(entry["abc_bytes"]),
        "activation_payload_already_read": bool(int(entry["pq_delta_dim"])),
        "q2_payload_already_read": True,
        "abc_payload_already_read": True,
        "q2_gate_up_scale_payload_already_read": bool(int(entry["pq_delta_dim"])),
        "activation_payload_bytes_read": ACTIVATION_PAYLOAD_BYTES if int(entry["pq_delta_dim"]) else 0,
        "q2_unit_feature_bytes_read": FP16_BYTES * UNITS * int(entry["q2_unit_dim"]),
        "abc_metadata_bytes_read": int(entry["abc_bytes"]),
    }
    return output, compute


def _charge_selector_score_payload(
    compute: Mapping[str, Any], *, units: int, abc_metadata_bytes: int,
    abc_score_required: bool = True,
) -> dict[str, Any]:
    result = dict(compute)
    if not bool(result["q2_payload_already_read"]):
        result["q2_unit_feature_bytes_read"] = 3 * int(units) * FP16_BYTES
        result["bytes_read"] = int(result["bytes_read"]) + int(result["q2_unit_feature_bytes_read"])
        result["q2_payload_already_read"] = True
    if not bool(result["abc_payload_already_read"]):
        result["abc_metadata_bytes_read"] = int(abc_metadata_bytes)
        result["bytes_read"] = int(result["bytes_read"]) + int(result["abc_metadata_bytes_read"])
        result["abc_payload_already_read"] = True
    expected_units = int(units) if bool(abc_score_required) else 0
    expected_macs = expected_units * ABC_SCORE_MACS_PER_UNIT
    if "abc_score_compute_macs" in result:
        if (
            int(result["abc_score_units"]) != expected_units
            or int(result["abc_score_compute_macs"]) != expected_macs
        ):
            raise RuntimeError("selector ABC-score accounting changed across repeated charging")
    else:
        result["abc_score_units"] = expected_units
        result["abc_score_compute_macs"] = expected_macs
        result["compute_macs"] = int(result.get("compute_macs", 0)) + expected_macs
    return result


def accounting_fields(
    config: Mapping[str, Any], *, expert_metadata_bytes: int, layer_shared_bytes: int,
    abc_bytes: int, compute_macs: int, compute_additions: int, selector_bytes_read: int,
    fetched_units: int, applied_units: int, scale_multiplications: int = 0,
    abc_score_units: int = 0, abc_score_compute_macs: int = 0,
) -> dict[str, Any]:
    if int(abc_score_compute_macs) != int(abc_score_units) * ABC_SCORE_MACS_PER_UNIT:
        raise ValueError("selector ABC-score MAC components are inconsistent")
    amortized = float(layer_shared_bytes) / float(config["experts_per_layer"])
    metadata = float(expert_metadata_bytes + abc_bytes) + amortized
    metadata_bpw = 8.0 * metadata / EXPERT_WEIGHTS
    physical_pages = 3 * int(fetched_units)
    physical_bytes = physical_pages * PAGE_BYTES
    logical_bytes = int(applied_units) * UNIT_PACKET_BYTES
    return {
        "physical_pages": physical_pages,
        "physical_bytes": physical_bytes,
        "physical_bpw": 8.0 * physical_bytes / EXPERT_WEIGHTS,
        "logical_actions": int(applied_units),
        "logical_bytes": logical_bytes,
        "logical_bpw": 8.0 * logical_bytes / EXPERT_WEIGHTS,
        "page_amplification": physical_bytes / max(logical_bytes, 1),
        "candidate_overfetch": int(fetched_units) / max(int(applied_units), 1),
        "expert_specific_metadata_bytes": int(expert_metadata_bytes),
        "layer_shared_metadata_bytes": int(layer_shared_bytes),
        "layer_shared_amortized_bytes_per_expert": amortized,
        "abc_metadata_bytes": int(abc_bytes),
        "selector_metadata_bytes_per_expert": metadata,
        "selector_metadata_bpw": metadata_bpw,
        "selector_compute_macs": int(compute_macs),
        "selector_abc_score_units": int(abc_score_units),
        "selector_abc_score_compute_macs": int(abc_score_compute_macs),
        "selector_abc_score_macs_per_unit": ABC_SCORE_MACS_PER_UNIT if int(abc_score_units) else 0,
        "selector_abc_score_mac_convention": ABC_SCORE_MAC_CONVENTION,
        "selector_compute_additions": int(compute_additions),
        "selector_scale_multiplications": int(scale_multiplications),
        "selector_bytes_read": int(selector_bytes_read),
        "selector_bytes_read_semantics": "logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic",
        "suffix_storage_bpw": float(config["suffix_bpw_per_complete_representation"]),
        "storage_multiplier": (
            float(config["reference_bpw"])
            + float(config["suffix_bpw_per_complete_representation"])
            + metadata_bpw
        ) / float(config["reference_bpw"]),
    }


def total_candidate_bytes_read(
    selector_bytes_read: int,
    rerank_incremental_bytes_read: int,
    fetched_units: int,
) -> int:
    """Combine reads without dropping or double-charging already-fetched packets."""
    fetched_packet_bytes = int(fetched_units) * UNIT_PACKET_BYTES
    return int(
        int(selector_bytes_read)
        + max(int(rerank_incremental_bytes_read) - fetched_packet_bytes, 0)
    )


def _require_positive_finite(value: float, label: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise RuntimeError(f"{label} must be finite and strictly positive, got {scalar!r}")
    return scalar


def _assert_gram_output_parity(
    outputs: Any, gram: np.ndarray, selected: Sequence[int],
    proxy: np.ndarray, beta: float,
) -> dict[str, float]:
    """Fail closed unless Gram damage equals direct output-residual qenergy."""
    matrix = np.asarray(gram, np.float64)
    ones = np.ones(outputs.units, np.float64)
    gram_base = _require_positive_finite(float(ones @ matrix @ ones), "Gram base damage")
    base_residual = np.asarray(outputs.target_output) - np.asarray(outputs.base_output)
    direct_base = _require_positive_finite(
        base.numpy_qenergy(base_residual, proxy=proxy, beta=beta),
        "direct base residual qenergy",
    )
    mask = np.zeros(outputs.units, bool)
    mask[np.asarray(selected, np.int64)] = True
    gram_selected = float(set_damage(matrix, mask))
    corrections = np.asarray(outputs.y11, np.float64) - np.asarray(outputs.y00, np.float64)
    direct_residual = base_residual - corrections[mask].sum(axis=0)
    direct_selected = float(base.numpy_qenergy(direct_residual, proxy=proxy, beta=beta))
    tolerance = 2e-5
    if not np.isclose(gram_base, direct_base, rtol=tolerance, atol=1e-8 * gram_base):
        raise RuntimeError(
            f"Gram/base output qenergy parity failed: {gram_base} != {direct_base}"
        )
    if not np.isfinite(gram_selected) or not np.isfinite(direct_selected):
        raise RuntimeError("selected Gram/output qenergy is non-finite")
    scale = max(abs(gram_selected), abs(direct_selected), 1e-20)
    if not np.isclose(gram_selected, direct_selected, rtol=tolerance, atol=1e-8 * gram_base):
        raise RuntimeError(
            f"Gram/selected output qenergy parity failed: {gram_selected} != {direct_selected}"
        )
    return {
        "gram_base_parity_relative_error": abs(gram_base - direct_base) / gram_base,
        "gram_selected_parity_relative_error": abs(gram_selected - direct_selected) / scale,
    }


def _validated_unit_ids(
    values: Sequence[int] | np.ndarray, *, label: str, units: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{label} must contain integer unit IDs")
    ids = raw.astype(np.int64, copy=False)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{label} must contain unique unit IDs")
    if np.any(ids < 0) or np.any(ids >= int(units)):
        raise ValueError(f"{label} contains an out-of-range unit ID")
    return ids


def candidate_rerank_records(
    gram: np.ndarray,
    candidates: Sequence[int],
    applied: int,
    *,
    independent_scores: Sequence[float],
    predicted_apply: Sequence[int] | None = None,
    activation_payload_already_read: bool,
    q2_payload_already_read: bool,
    abc_payload_already_read: bool,
    abc_metadata_bytes: int,
    proxy_rank: int,
    q2_gate_up_scale_payload_already_read: bool = False,
) -> list[dict[str, Any]]:
    """Audit four interfaces with explicit information and compute semantics."""
    matrix = np.asarray(gram, np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(~np.isfinite(matrix)):
        raise ValueError("gram must be a finite square matrix")
    units = int(matrix.shape[0])
    applied_count = int(applied)
    if applied_count <= 0 or applied_count > units:
        raise ValueError("applied must be positive and no greater than the unit count")
    if int(abc_metadata_bytes) < 0:
        raise ValueError("abc_metadata_bytes must be nonnegative")
    metric_proxy_rank = int(proxy_rank)
    if metric_proxy_rank < 0:
        raise ValueError("proxy_rank must be nonnegative")
    candidate_ids = _validated_unit_ids(
        candidates, label="candidate_unit_ids", units=units,
    )
    if len(candidate_ids) < applied_count:
        raise ValueError("candidate support must contain at least applied units")
    predicted_values = (
        candidate_ids[:applied_count] if predicted_apply is None else predicted_apply
    )
    predicted = _validated_unit_ids(
        predicted_values, label="predicted_apply", units=units,
    )
    candidate_set = set(map(int, candidate_ids))
    if len(predicted) != applied_count:
        raise ValueError("predicted apply support must have exactly applied unit IDs")
    if not set(map(int, predicted)) <= candidate_set:
        raise ValueError("predicted apply support must be a candidate subset")

    score = np.asarray(independent_scores, np.float64).reshape(-1)
    if score.shape != (units,) or np.any(~np.isfinite(score)):
        raise ValueError("independent_scores must be one finite decoded-ABC score per unit")
    started = time.perf_counter()
    independent = candidate_ids[
        np.lexsort((candidate_ids, -score[candidate_ids]))
    ][: int(applied)]
    independent_runtime = time.perf_counter() - started

    started = time.perf_counter()
    contained = exact_contained_target_fixed_greedy(matrix, candidate_ids, applied_count)
    contained_runtime = time.perf_counter() - started
    started = time.perf_counter()
    full = exact_full_target_fixed_greedy(matrix, applied_count, candidates=candidate_ids)
    full_runtime = time.perf_counter() - started

    base_damage = _require_positive_finite(
        float(np.ones(units) @ matrix @ np.ones(units)),
        "candidate base damage",
    )
    oracle = exact_full_target_fixed_greedy(matrix, applied_count)
    oracle_mask = np.zeros(units, bool)
    oracle_mask[np.asarray(oracle.order, np.int64)] = True
    oracle_gain = _require_positive_finite(
        set_gain(matrix, oracle_mask), "candidate oracle set gain",
    )
    oracle_ids = _validated_unit_ids(
        oracle.order, label="oracle_selected_units", units=units,
    )
    if len(oracle_ids) != applied_count:
        raise RuntimeError("oracle support cardinality differs from applied count")
    fetched = len(candidate_ids)
    activation_incremental_bytes = 0 if bool(activation_payload_already_read) else ACTIVATION_PAYLOAD_BYTES
    q2_incremental_bytes = 0 if bool(q2_payload_already_read) else 3 * units * FP16_BYTES
    abc_incremental_bytes = 0 if bool(abc_payload_already_read) else int(abc_metadata_bytes)
    proxy_metadata_bytes = INPUTS * metric_proxy_rank * FP32_BYTES
    q2_parent_code_bytes = fetched * Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT
    q2_scale_bytes = (
        0 if bool(q2_gate_up_scale_payload_already_read)
        else fetched * Q2_GATE_UP_SCALE_BYTES_PER_UNIT
    )
    q4_response_macs = 2 * fetched * INPUTS
    down_correction_macs = 2 * fetched * INPUTS
    correction_workspace_bytes = fetched * INPUTS * FP16_BYTES
    interaction_euclidean_gram_macs = fetched * fetched * INPUTS
    interaction_proxy_projection_macs = fetched * INPUTS * metric_proxy_rank
    interaction_proxy_gram_macs = fetched * fetched * metric_proxy_rank
    interaction_gram_macs = (
        interaction_euclidean_gram_macs
        + interaction_proxy_projection_macs
        + interaction_proxy_gram_macs
    )
    full_q4_response_macs = 2 * units * INPUTS
    full_down_correction_macs = 2 * units * INPUTS
    full_correction_workspace_bytes = units * INPUTS * FP16_BYTES
    full_interaction_euclidean_gram_macs = units * units * INPUTS
    full_q2_parent_code_bytes = units * Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT
    full_q2_scale_bytes = (
        0 if bool(q2_gate_up_scale_payload_already_read)
        else units * Q2_GATE_UP_SCALE_BYTES_PER_UNIT
    )
    full_interaction_proxy_projection_macs = units * INPUTS * metric_proxy_rank
    full_interaction_proxy_gram_macs = units * units * metric_proxy_rank
    full_interaction_gram_macs = (
        full_interaction_euclidean_gram_macs
        + full_interaction_proxy_projection_macs
        + full_interaction_proxy_gram_macs
    )
    selections = (
        (
            "predicted_direct_no_rerank", predicted, "deployable_predicted_application",
            0, 0, 0.0, "primary_predictor_output", 0, 0, 0, False,
        ),
        (
            "exact_independent_abc_within_fetched_candidates", independent,
            "deployable_after_candidate_fetch",
            q4_response_macs + ABC_SCORE_MACS_PER_UNIT * fetched,
            fetched * UNIT_PACKET_BYTES
            + activation_incremental_bytes
            + q2_incremental_bytes
            + abc_incremental_bytes
            + q2_parent_code_bytes
            + q2_scale_bytes,
            independent_runtime, "primary_realistic_h0_rerank", fetched, 0, 0, False,
        ),
        (
            "contained_fetched_target_exact_h0", np.asarray(contained.order, np.int64),
            "information_deployable_but_expensive_interaction_oracle",
            q4_response_macs + down_correction_macs + interaction_gram_macs,
            fetched * UNIT_PACKET_BYTES
            + activation_incremental_bytes
            + q2_incremental_bytes
            + q2_parent_code_bytes
            + q2_scale_bytes
            + correction_workspace_bytes
            + fetched * fetched * FP32_BYTES
            + proxy_metadata_bytes,
            contained_runtime, "systems_oracle_not_promotion_primary",
            fetched, correction_workspace_bytes, fetched, True,
        ),
        (
            "full_target_restricted_teacher_oracle", np.asarray(full.order, np.int64),
            "teacher_oracle_not_deployable",
            full_q4_response_macs + full_down_correction_macs + full_interaction_gram_macs,
            units * UNIT_PACKET_BYTES
            + activation_incremental_bytes
            + q2_incremental_bytes
            + full_q2_parent_code_bytes
            + full_q2_scale_bytes
            + full_correction_workspace_bytes
            + units * units * FP32_BYTES
            + proxy_metadata_bytes,
            full_runtime, "teacher_information_fixed_greedy_control",
            units, full_correction_workspace_bytes, units, True,
        ),
    )
    records = []
    for (
        semantics, selected, regime, macs, bytes_read, runtime, role,
        q4_units, workspace_bytes, gram_units, lower_bound,
    ) in selections:
        selected_ids = _validated_unit_ids(
            selected, label=f"selected_units[{semantics}]", units=units,
        )
        if len(selected_ids) != applied_count:
            raise RuntimeError(f"{semantics} support cardinality differs from applied count")
        if not set(map(int, selected_ids)) <= candidate_set:
            raise RuntimeError(f"{semantics} selected a unit outside fetched candidates")
        mask = np.zeros(units, bool)
        mask[selected_ids] = True
        gain = set_gain(matrix, mask)
        is_independent = semantics == "exact_independent_abc_within_fetched_candidates"
        has_interaction = int(gram_units) > 0
        row_activation_bytes = activation_incremental_bytes if int(q4_units) else 0
        row_q2_bytes = q2_incremental_bytes if int(q4_units) else 0
        row_abc_bytes = abc_incremental_bytes if is_independent else 0
        row_q2_parent_code_bytes = (
            int(q4_units) * Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT
        )
        row_q2_scale_bytes = (
            0 if bool(q2_gate_up_scale_payload_already_read)
            else int(q4_units) * Q2_GATE_UP_SCALE_BYTES_PER_UNIT
        )
        row_down_correction_macs = 2 * int(gram_units) * INPUTS
        row_euclidean_gram_macs = int(gram_units) * int(gram_units) * INPUTS
        row_proxy_projection_macs = int(gram_units) * INPUTS * metric_proxy_rank
        row_proxy_gram_macs = int(gram_units) * int(gram_units) * metric_proxy_rank
        row_interaction_gram_macs = row_euclidean_gram_macs + row_proxy_projection_macs + row_proxy_gram_macs
        row_proxy_metadata_bytes = proxy_metadata_bytes if has_interaction else 0
        row_abc_macs_per_unit = ABC_SCORE_MACS_PER_UNIT if is_independent else 0
        expected_macs = (
            2 * int(q4_units) * INPUTS
            + row_abc_macs_per_unit * int(q4_units)
            + row_down_correction_macs
            + row_interaction_gram_macs
        )
        if int(macs) != int(expected_macs):
            raise RuntimeError(
                f"{semantics} rerank MAC total disagrees with component charges"
            )
        expected_bytes = (
            int(q4_units) * UNIT_PACKET_BYTES
            + row_activation_bytes
            + row_q2_bytes
            + row_abc_bytes
            + int(workspace_bytes)
            + row_q2_parent_code_bytes
            + row_q2_scale_bytes
            + int(gram_units) * int(gram_units) * FP32_BYTES
            + row_proxy_metadata_bytes
        )
        if int(bytes_read) != int(expected_bytes):
            raise RuntimeError(
                f"{semantics} rerank byte total disagrees with component charges"
            )
        records.append({
            "rerank_semantics": semantics,
            "rerank_role": role,
            "selection_regime": regime,
            "recovery": 1.0 - set_damage(matrix, mask) / base_damage,
            "set_gain": gain,
            "oracle_set_gain": float(oracle_gain),
            "oracle_set_gain_retention": gain / oracle_gain,
            "candidate_unit_ids": json.dumps(list(map(int, candidate_ids))),
            "oracle_selected_units": json.dumps(list(map(int, oracle_ids))),
            "selected_units": json.dumps(list(map(int, selected_ids))),
            "rerank_incremental_compute_macs": int(macs),
            "rerank_incremental_bytes_read": int(bytes_read),
            "activation_payload_already_read": bool(activation_payload_already_read),
            "q2_payload_already_read": bool(q2_payload_already_read),
            "abc_payload_already_read": bool(abc_payload_already_read),
            "rerank_activation_payload_bytes_read": int(row_activation_bytes),
            "rerank_q2_unit_feature_bytes_read": int(row_q2_bytes),
            "rerank_abc_metadata_bytes_read": int(row_abc_bytes),
            "rerank_candidate_packet_bytes_read": int(q4_units) * UNIT_PACKET_BYTES,
            "rerank_candidate_packet_contents": CANDIDATE_PACKET_CONTENTS,
            "rerank_resident_q2_down_payload_required": bool(has_interaction),
            "rerank_q2_gate_up_parent_payload_required": bool(q4_units),
            "rerank_q2_gate_up_parent_payload_accounted": True,
            "rerank_q2_gate_up_parent_code_bytes_read": int(row_q2_parent_code_bytes),
            "rerank_q2_gate_up_scale_bytes_read": int(row_q2_scale_bytes),
            "rerank_q2_gate_up_scale_payload_already_read": bool(
                q2_gate_up_scale_payload_already_read
            ),
            "rerank_q4_response_scale_multiplications": int(q4_units) * Q4_RESPONSE_SCALE_MULTIPLICATIONS_PER_UNIT,
            "rerank_resident_q2_down_payload_accounted": not bool(has_interaction),
            "rerank_packet_bytes_already_in_selector_bytes_read": True,
            "rerank_q4_response_units": int(q4_units),
            "rerank_q4_response_compute_macs": int(2 * q4_units * INPUTS),
            "rerank_down_correction_compute_macs": int(row_down_correction_macs),
            "rerank_correction_workspace_bytes": int(workspace_bytes),
            "rerank_interaction_gram_units": int(gram_units),
            "rerank_interaction_gram_euclidean_compute_macs": int(row_euclidean_gram_macs),
            "rerank_interaction_proxy_rank": int(metric_proxy_rank if has_interaction else 0),
            "rerank_interaction_proxy_projection_compute_macs": int(row_proxy_projection_macs),
            "rerank_interaction_proxy_gram_compute_macs": int(row_proxy_gram_macs),
            "rerank_interaction_proxy_scale_multiplications": int(gram_units) * int(gram_units) if metric_proxy_rank else 0,
            "rerank_proxy_metadata_bytes_read": int(row_proxy_metadata_bytes),
            "rerank_interaction_gram_compute_macs": int(row_interaction_gram_macs),
            "rerank_abc_score_macs_per_unit": int(row_abc_macs_per_unit),
            "rerank_abc_score_mac_convention": ABC_SCORE_MAC_CONVENTION,
            "rerank_non_mac_operations": RERANK_NON_MAC_CONVENTION,
            "rerank_accounting_is_lower_bound": bool(lower_bound),
            "rerank_accounting_bound": (
                INTERACTION_RERANK_ACCOUNTING_BOUND
                if lower_bound else "complete_under_declared_logical_payload_and_MAC_conventions"
            ),
            "rerank_unaccounted_overhead": (
                INTERACTION_RERANK_UNACCOUNTED
                if lower_bound else "none_under_declared_logical_payload_and_MAC_conventions"
            ),
            "rerank_reference_runtime_ms": 1000.0 * float(runtime),
            "rerank_score_semantics": (
                INDEPENDENT_ABC_SCORE_SEMANTICS
                if semantics == "exact_independent_abc_within_fetched_candidates"
                else "not_independent_abc"
            ),
        })
    return records


def _factorized_states(outputs: Any, chosen: Sequence[int]) -> np.ndarray:
    states = np.zeros(outputs.units, np.int64)
    for action in map(int, chosen):
        block, unit = divmod(action, outputs.units)
        if block == 0:
            states[unit] = 1
        elif block == 1:
            states[unit] = 2
        elif block in (2, 3):
            states[unit] = STATE_GD
        else:
            raise RuntimeError("PR9 factorized path returned an unknown transition block")
    return states


def _best_hybrid_forward_prefix(
    outputs: Any, trace: Any, proxy: np.ndarray, beta: float,
) -> dict[str, Any]:
    """Exactly reprice every cumulative prefix, including the empty state."""
    states = np.zeros(outputs.units, np.int64)
    contributions = np.stack(
        (outputs.y00, outputs.y10, outputs.y01, outputs.y11),
    ).astype(np.float64, copy=False)
    current = np.asarray(outputs.base_output, np.float64).copy()
    target = np.asarray(outputs.target_output, np.float64)
    best_damage = float(base.numpy_qenergy(target - current, proxy=proxy, beta=beta))
    best_states = states.copy()
    best_output = current.copy()
    best_pages = 0
    best_actions = 0
    for prefix, action in enumerate(trace.actions, start=1):
        unit = int(action.unit)
        if int(states[unit]) != int(action.from_state):
            raise RuntimeError("hybrid forward trace is not a valid state path")
        delta = (
            contributions[int(action.to_state), unit]
            - contributions[int(action.from_state), unit]
        )
        states[unit] = int(action.to_state)
        current += delta
        damage = float(base.numpy_qenergy(target - current, proxy=proxy, beta=beta))
        if (
            damage < best_damage - 1e-12
            or (
                abs(damage - best_damage) <= 1e-12
                and int(action.cumulative_pages) < best_pages
            )
        ):
            best_damage = damage
            best_states = states.copy()
            best_output = current.copy()
            best_pages = int(action.cumulative_pages)
            best_actions = prefix
    return {
        "states": best_states,
        "output": best_output,
        "residual_energy": best_damage,
        "pages": best_pages,
        "actions": best_actions,
        "terminal_actions": len(trace.actions),
        "terminal_pages": int(trace.pages),
    }


def _hybrid_rows(
    outputs: Any, gram: np.ndarray, proxy: np.ndarray, beta: float,
    independent_abc_score: np.ndarray,
    metadata: Mapping[str, Any], config: Mapping[str, Any],
    device: torch.device | str,
) -> list[dict[str, Any]]:
    """Evaluate one shared GPU base-Gram path across a bounded budget grid."""
    rows: list[dict[str, Any]] = []
    base_damage = _require_positive_finite(
        float(np.ones(len(gram)) @ gram @ np.ones(len(gram))),
        "hybrid base damage",
    )
    independent_score = np.asarray(independent_abc_score, np.float64)
    if independent_score.shape != (outputs.units,):
        raise ValueError("decoded ABC independent score width changed")
    independent_order = np.lexsort((np.arange(len(independent_score)), -independent_score))
    budgets = tuple(map(float, config["hybrid_budgets_bpw"]))
    page_budgets = tuple(int(round(value * PAGES_PER_BPW)) for value in budgets)

    accelerated = exact_hybrid_unit_base_gram_greedy(
        outputs, page_budgets, proxy=proxy, beta=beta,
        device=device, dtype=torch.float32,
    )
    factorized_started = time.perf_counter()
    factorized_trace = prior.exact_factorized_neuron_fixed_greedy(
        outputs, page_budgets=page_budgets, proxy=proxy, beta=beta, device=device,
    )
    factorized_shared_runtime = time.perf_counter() - factorized_started

    for budget_bpw, pages in zip(budgets, page_budgets, strict=True):
        coherent_count = min(outputs.units, pages // 3)

        started = time.perf_counter()
        independent_states = np.zeros(outputs.units, np.int64)
        independent_states[independent_order[:coherent_count]] = STATE_GD
        independent_damage = set_damage(gram, independent_states == STATE_GD)
        independent_runtime = time.perf_counter() - started

        started = time.perf_counter()
        teacher = exact_full_target_fixed_greedy(gram, coherent_count)
        coherent_states = np.zeros(outputs.units, np.int64)
        coherent_states[teacher.order] = STATE_GD
        coherent_damage = set_damage(gram, coherent_states == STATE_GD)
        coherent_runtime = time.perf_counter() - started

        started = time.perf_counter()
        factorized_output, _, factorized_pages, factorized_chosen, _ = prior.best_factorized_prefix(
            outputs, factorized_trace, pages,
        )
        factorized_states = _factorized_states(outputs, factorized_chosen)
        factorized_residual = np.asarray(outputs.target_output) - factorized_output
        factorized_damage = base.numpy_qenergy(
            factorized_residual, proxy=proxy, beta=beta,
        )
        factorized_snapshot_runtime = time.perf_counter() - started

        forward_snapshot = accelerated.snapshots[int(pages)]
        forward_states = np.asarray(forward_snapshot.states, np.int64)

        started = time.perf_counter()
        local = exact_monotone_unit_local_search(
            outputs, {
                "coherent_independent_abc_pr9": independent_states,
                "coherent_exact_set": coherent_states,
                "direct_GD_hybrid_best_prefix": forward_states,
                "pr9_factorized_fixed_greedy": factorized_states,
            }, pages,
            shortlist_size=int(config["hybrid_local_shortlist"]),
            max_swap_units=int(config["hybrid_local_swap_units"]),
            max_passes=int(config["hybrid_local_max_passes"]),
            page_balanced=False,
            proxy=proxy, beta=beta,
        )
        local_incremental_runtime = time.perf_counter() - started
        seed_runtime = (
            independent_runtime + coherent_runtime + factorized_shared_runtime
            + factorized_snapshot_runtime + accelerated.total_seconds
        )

        configurations = (
            (
                "coherent_independent_abc_pr9_control",
                independent_damage, 3 * coherent_count, "independent_abc",
                independent_runtime, independent_runtime,
                "PR9 decoded-FP16 ABC independent correction-norm order",
            ),
            (
                "coherent_exact_set_fixed_greedy_teacher",
                coherent_damage, 3 * coherent_count, "coherent_exact_set",
                coherent_runtime, coherent_runtime,
                "residual-aware fixed greedy; not a global subset optimum",
            ),
            (
                "factorized_fixed_greedy_pr9_control",
                factorized_damage, factorized_pages, "pr9_factorized_fixed_greedy",
                factorized_snapshot_runtime,
                factorized_shared_runtime + factorized_snapshot_runtime,
                "PR9 factorized best profitable prefix from one shared GPU path",
            ),
            (
                "direct_GD_hybrid_forward",
                forward_snapshot.residual_energy, forward_snapshot.pages,
                "direct_GD_hybrid_best_prefix", 0.0, accelerated.total_seconds,
                "best direct-qenergy prefix of one shared GPU primitive-Gram path",
            ),
            (
                "hybrid_forward_plus_1_2_3_unit_local_search",
                local.residual_energy, local.pages, local.seed_name,
                local_incremental_runtime, seed_runtime + local_incremental_runtime,
                "monotone exact add/drop/swap search over four named seed basins",
            ),
        )
        for (
            family, damage, used_pages, seed, incremental_runtime, total_runtime, semantics,
        ) in configurations:
            physical_bytes = int(used_pages) * PAGE_BYTES
            rows.append({
                **metadata,
                "selector_family": family,
                "selector_semantics": semantics,
                "selection_regime": "h0_exact_oracle",
                "physical_budget_bpw": budget_bpw,
                "physical_pages": int(used_pages),
                "physical_bytes": physical_bytes,
                "physical_bpw": 8.0 * physical_bytes / EXPERT_WEIGHTS,
                "page_amplification": 1.0,
                "recovery": 1.0 - float(damage) / base_damage,
                "selected_seed": seed,
                "selector_runtime_ms": 1000.0 * float(total_runtime),
                "selector_runtime_incremental_ms": 1000.0 * float(incremental_runtime),
                "selector_runtime_total_including_seed_ms": 1000.0 * float(total_runtime),
                "physical_budget_pages": pages,
                "unused_budget_pages": pages - int(used_pages),
                "hybrid_forward_best_prefix_actions": int(forward_snapshot.best_prefix),
                "hybrid_forward_terminal_actions": len(accelerated.actions),
                "hybrid_forward_terminal_pages": int(accelerated.terminal_pages),
                "hybrid_forward_profitable": bool(forward_snapshot.profitable),
                "local_search_page_balanced": False,
                "hybrid_reference_backend": "torch_gpu_primitive_base_gram_cpu_correlation_update",
                "hybrid_gram_device": str(accelerated.gram_device),
                "hybrid_gram_dtype": str(accelerated.gram_dtype),
                "hybrid_primitive_gram_device_bytes": int(
                    accelerated.primitive_gram_device_bytes
                ),
                "hybrid_candidate_evaluations": int(accelerated.candidate_evaluations),
                "hybrid_gram_build_ms": 1000.0 * float(accelerated.gram_build_seconds),
                "hybrid_gram_transfer_ms": 1000.0 * float(accelerated.gram_transfer_seconds),
                "hybrid_greedy_ms": 1000.0 * float(accelerated.greedy_seconds),
                "hybrid_snapshot_ms": 1000.0 * float(accelerated.snapshot_seconds),
                "hybrid_shared_grid_total_ms": 1000.0 * float(accelerated.total_seconds),
                "runtime_shared_across_budget_grid": family in {
                    "factorized_fixed_greedy_pr9_control", "direct_GD_hybrid_forward",
                },
            })
    return rows


def _selector_prediction(
    arrays: Mapping[str, np.ndarray], entry: Mapping[str, Any], activation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    family = str(entry["selector_family"])
    if family == "block_pq_residual_synopsis":
        gate_encoded = _pq_encoded(arrays, entry, "gate")
        up_encoded = _pq_encoded(arrays, entry, "up")
        gate = evaluate_block_pq_response_lut(gate_encoded, activation, activation_bytes_per_value=2)
        up = evaluate_block_pq_response_lut(up_encoded, activation, activation_bytes_per_value=2)
        shared_bytes = int(entry["layer_shared_bytes"])
        macs = int(
            gate.accounting.codebook_lut_macs + up.accounting.codebook_lut_macs
        )
        additions = int(
            gate.accounting.response_accumulation_additions
            + up.accounting.response_accumulation_additions
        )
        scale_multiplications = int(
            gate.accounting.scale_multiplications + up.accounting.scale_multiplications
        )
        bytes_read = int(
            shared_bytes + int(entry["expert_specific_bytes"])
            + gate.accounting.block_scale_bytes_read + up.accounting.block_scale_bytes_read
            + gate.accounting.activation_bytes
            + gate.accounting.lut_read_bytes + up.accounting.lut_read_bytes
        )
        return gate.response, up.response, {
            "compute_macs": macs,
            "compute_additions": additions,
            "scale_multiplications": scale_multiplications,
            "bytes_read": bytes_read,
            "activation_payload_already_read": True,
            "q2_payload_already_read": False,
            "abc_payload_already_read": False,
            "q2_gate_up_scale_payload_already_read": True,
            "activation_payload_bytes_read": int(gate.accounting.activation_bytes),
            "q2_unit_feature_bytes_read": 0,
            "abc_metadata_bytes_read": 0,
            "runtime_implementation": "reference_evaluates_two_projection_LUT_banks",
            "analytical_contract": "activation_read_once_separate_gate_up_LUT_banks",
        }
    if family == "high_rank_low_bit_linear_response":
        analysis = np.asarray(arrays[str(entry["analysis_key"])], np.float32)
        latent = analysis @ np.asarray(activation, np.float32)
        gate_matrix = _high_rank_matrix(arrays, entry, "gate")
        up_matrix = _high_rank_matrix(arrays, entry, "up")
        gate, up = gate_matrix @ latent, up_matrix @ latent
        rank = int(entry["rank"])
        return gate, up, {
            "compute_macs": int(INPUTS * rank + 2 * UNITS * rank),
            "compute_additions": 0,
            "scale_multiplications": 2 * UNITS,
            "bytes_read": int(entry["layer_shared_bytes"] + entry["expert_specific_bytes"] + 2 * INPUTS),
            "activation_payload_already_read": True,
            "q2_payload_already_read": False,
            "abc_payload_already_read": False,
            "q2_gate_up_scale_payload_already_read": False,
            "activation_payload_bytes_read": ACTIVATION_PAYLOAD_BYTES,
            "q2_unit_feature_bytes_read": 0,
            "abc_metadata_bytes_read": 0,
            "runtime_implementation": "dense_decoded_reference",
            "analytical_contract": "packed_low_bit_rows_with_fp16_row_scales",
        }
    raise RuntimeError(f"unknown selector family: {family}")


def runtime_independent_abc_scores(
    q2_gate: np.ndarray,
    q2_up: np.ndarray,
    q4_gate: np.ndarray,
    q4_up: np.ndarray,
    activation: np.ndarray,
    encoded_metadata: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Score coherent units through the declared H0 FP16-rounded runtime path.

    The reference evaluates all unit responses for convenience. Candidate
    accounting continues to charge only the fetched units that are eligible
    for the primary independent-ABC rerank.
    """
    runtime_activation = np.asarray(activation, np.float16).astype(np.float32)
    gate2 = np.asarray(
        np.asarray(q2_gate, np.float32) @ runtime_activation,
        np.float16,
    ).astype(np.float32)
    up2 = np.asarray(
        np.asarray(q2_up, np.float32) @ runtime_activation,
        np.float16,
    ).astype(np.float32)
    runtime_h2 = np.asarray(silu(gate2) * up2, np.float16).astype(np.float32)
    exact_delta_gate = np.asarray(
        (np.asarray(q4_gate, np.float32) - np.asarray(q2_gate, np.float32))
        @ runtime_activation,
        np.float32,
    )
    exact_delta_up = np.asarray(
        (np.asarray(q4_up, np.float32) - np.asarray(q2_up, np.float32))
        @ runtime_activation,
        np.float32,
    )
    exact_runtime_h4 = silu(gate2 + exact_delta_gate) * (up2 + exact_delta_up)
    score = complete_unit_scores(runtime_h2, exact_runtime_h4, encoded_metadata)
    return score, {
        "runtime_activation": runtime_activation,
        "gate2": gate2,
        "up2": up2,
        "runtime_h2": runtime_h2,
        "exact_delta_gate": exact_delta_gate,
        "exact_delta_up": exact_delta_up,
        "exact_runtime_h4": np.asarray(exact_runtime_h4, np.float32),
    }


def evaluate_invocation(
    matrices: Mapping[str, list[np.ndarray]], activation: np.ndarray,
    proxy: np.ndarray, beta: float, static_down_metric: np.ndarray,
    encoded_metadata: Any, abc_bytes: int, metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray], layer_record: Mapping[str, Any],
    config: Mapping[str, Any], *, run_hybrid_oracle: bool,
    hybrid_device: torch.device | str,
) -> dict[str, list[dict[str, Any]]]:
    q2 = tuple(matrices[name][0] for name in PROJECTIONS)
    q4 = tuple(matrices[name][2] for name in PROJECTIONS)
    proxy_value = np.asarray(proxy)
    if proxy_value.ndim == 1 and proxy_value.shape[0] == INPUTS:
        metric_proxy_rank = 1
    elif proxy_value.ndim == 2 and proxy_value.shape[0] == INPUTS:
        metric_proxy_rank = int(proxy_value.shape[1])
    else:
        raise ValueError("future proxy must have shape [2048] or [2048, rank]")
    if metric_proxy_rank <= 0:
        raise ValueError("future proxy rank must be positive")
    outputs = factorized_unit_outputs(q2, q4, activation)
    gram = unit_correction_gram(outputs.h2, outputs.h4, static_down_metric)
    exact_path = exact_full_target_fixed_greedy(gram, int(config["teacher_path_length"]))
    parity = _assert_gram_output_parity(
        outputs, gram, exact_path.order[: int(config["applied_units"])], proxy, beta,
    )
    metadata = {**metadata, **parity, "gram_output_qenergy_parity_asserted": True}
    path_utility = np.zeros(outputs.units, np.float64)
    apply_prefix = int(config["applied_units"])
    path_utility[exact_path.order[:apply_prefix]] = np.maximum(
        exact_path.marginal_gain[:apply_prefix], 0.0,
    )
    independent_abc_score, runtime = runtime_independent_abc_scores(
        q2[0], q2[1], q4[0], q4[1], activation, encoded_metadata,
    )
    result = {name: [] for name in ARTIFACTS}
    if run_hybrid_oracle:
        result["hybrid_oracle_frontier"].extend(
            _hybrid_rows(
                outputs, gram, proxy, beta, independent_abc_score, metadata, config,
                device=hybrid_device,
            )
        )

    template_entries = [
        entry for entry in layer_record["template_entries"]
        if int(entry["expert_id"]) == int(metadata["expert_id"])
    ]
    for entry in template_entries:
        for requested in map(int, entry["requested_template_counts"]):
            templates = _template_masks(arrays, entry, requested)
            if not len(templates):
                continue
            template_bank_metadata_bytes = int(len(templates) * ((UNITS + 7) // 8))
            template_metadata_bpw = 8.0 * template_bank_metadata_bytes / EXPERT_WEIGHTS
            template_storage_multiplier = (
                float(config["reference_bpw"])
                + float(config["suffix_bpw_per_complete_representation"])
                + template_metadata_bpw
            ) / float(config["reference_bpw"])
            for union_name, selector in (
                ("top1_template_validation_oracle", charged_top1_template_candidates),
                ("top2_union_validation_oracle", charged_top2_union_candidates),
            ):
                for repair_count in map(int, config["template_repair_counts"]):
                    candidate = selector(
                        templates, path_utility, repair_count=repair_count,
                        pages_per_unit=3, page_bytes=PAGE_BYTES,
                        applied_units=int(config["applied_units"]),
                    )
                    ids = np.asarray(candidate.candidate_units, np.int64)
                    candidate_cap_pass = len(ids) <= int(config["candidate_units"])
                    result["template_frontier"].append({
                        **metadata,
                        "template_cohort": str(entry["cohort"]),
                        "pairing_semantics": str(entry["pairing_semantics"]),
                        "template_selector": union_name,
                        "template_count_requested": requested,
                        "template_count_effective": len(templates),
                        "repair_count": repair_count,
                        "repair_semantics": "exact_validation_path_utility_oracle_not_deployable_classifier",
                        "candidate_units": len(ids),
                        "candidate_cap_units": int(config["candidate_units"]),
                        "candidate_cap_pass": candidate_cap_pass,
                        "physical_pages": candidate.pages,
                        "physical_bytes": candidate.bytes_read,
                        "physical_bpw": 8.0 * candidate.bytes_read / EXPERT_WEIGHTS,
                        "logical_actions": int(config["applied_units"]),
                        "logical_bytes": int(config["applied_units"]) * UNIT_PACKET_BYTES,
                        "logical_bpw": 3 * int(config["applied_units"]) / PAGES_PER_BPW,
                        "page_amplification": len(ids) / int(config["applied_units"]),
                        "candidate_overfetch": candidate.overfetch,
                        "validation_oracle_path_utility_retained": candidate.retained_utility,
                        "selection_regime": "h0_validation_template_oracle_not_classifier",
                        "template_indices": json.dumps(list(candidate.template_indices)),
                        "template_bank_metadata_bytes": template_bank_metadata_bytes,
                        "template_classifier_metadata_bytes": 0,
                        "selector_metadata_bytes_per_expert": template_bank_metadata_bytes,
                        "selector_metadata_bpw": template_metadata_bpw,
                        "storage_multiplier": template_storage_multiplier,
                        "selector_metadata_semantics": (
                            "stored_mask_bank_only; validation-oracle template/repair choice "
                            "has no deployable classifier"
                        ),
                        "promotable": False,
                    })
                    if len(ids) >= int(config["applied_units"]):
                        predicted = ids[
                            np.lexsort((ids, -path_utility[ids]))
                        ][: int(config["applied_units"])]
                        for row in candidate_rerank_records(
                            gram, ids, int(config["applied_units"]),
                            independent_scores=independent_abc_score,
                            predicted_apply=predicted,
                            activation_payload_already_read=False,
                            q2_payload_already_read=False,
                            abc_payload_already_read=False,
                            abc_metadata_bytes=int(abc_bytes),
                            proxy_rank=metric_proxy_rank,
                            q2_gate_up_scale_payload_already_read=False,
                        ):
                            result["candidate_frontier"].append({
                                **metadata,
                                **row,
                                "selector_config_id": (
                                    f"template:{entry['cohort']}:K{requested}:"
                                    f"{union_name}:repair{repair_count}"
                                ),
                                "selector_family": "support_template",
                                "template_cohort": str(entry["cohort"]),
                                "repair_count": repair_count,
                                "candidate_units": len(ids),
                                "candidate_cap_units": int(config["candidate_units"]),
                                "candidate_cap_pass": candidate_cap_pass,
                                "applied_units": int(config["applied_units"]),
                                "physical_pages": candidate.pages,
                                "physical_bytes": candidate.bytes_read,
                                "physical_bpw": 8.0 * candidate.bytes_read / EXPERT_WEIGHTS,
                                "logical_actions": int(config["applied_units"]),
                                "logical_bytes": int(config["applied_units"]) * UNIT_PACKET_BYTES,
                                "logical_bpw": 3 * int(config["applied_units"]) / PAGES_PER_BPW,
                                "page_amplification": len(ids) / int(config["applied_units"]),
                                "candidate_overfetch": len(ids) / int(config["applied_units"]),
                                "template_bank_metadata_bytes": template_bank_metadata_bytes,
                                "template_classifier_metadata_bytes": 0,
                                "selector_metadata_bytes_per_expert": template_bank_metadata_bytes,
                                "selector_metadata_bpw": template_metadata_bpw,
                                "storage_multiplier": template_storage_multiplier,
                                "selector_metadata_semantics": (
                                    "stored_mask_bank_only; exact validation path utility "
                                    "chooses template/repair/apply and is not a classifier"
                                ),
                                "selector_compute_macs": 0,
                                "selector_abc_score_units": 0,
                                "selector_abc_score_compute_macs": 0,
                                "selector_abc_score_macs_per_unit": 0,
                                "selector_abc_score_mac_convention": ABC_SCORE_MAC_CONVENTION,
                                "selector_bytes_read": (
                                    template_bank_metadata_bytes + candidate.bytes_read
                                ),
                                "selection_regime": (
                                    "h0_validation_template_oracle_then_"
                                    + str(row["rerank_semantics"])
                                ),
                                "rerank_role": (
                                    "template_validation_oracle_nonpromotable__"
                                    + str(row["rerank_role"])
                                ),
                                "promotable": False,
                                "total_selector_rerank_compute_macs": int(
                                    row["rerank_incremental_compute_macs"]
                                ),
                                "total_selector_rerank_bytes_read": int(
                                    total_candidate_bytes_read(
                                        template_bank_metadata_bytes + candidate.bytes_read,
                                        row["rerank_incremental_bytes_read"],
                                        len(ids),
                                    )
                                ),
                                "total_selector_compute_macs": int(
                                    row["rerank_incremental_compute_macs"]
                                ),
                                "total_selector_bytes_read": int(
                                    total_candidate_bytes_read(
                                        template_bank_metadata_bytes + candidate.bytes_read,
                                        row["rerank_incremental_bytes_read"],
                                        len(ids),
                                    )
                                ),
                            })

    runtime_activation = runtime["runtime_activation"]
    gate2 = runtime["gate2"]
    up2 = runtime["up2"]
    runtime_h2 = runtime["runtime_h2"]
    exact_delta_gate = runtime["exact_delta_gate"]
    exact_delta_up = runtime["exact_delta_up"]
    exact_runtime_h4 = runtime["exact_runtime_h4"]
    abc_features = np.stack(
        (encoded_metadata.a, encoded_metadata.b, encoded_metadata.c), axis=1,
    ).astype(np.float32)
    expert_id = int(metadata["expert_id"])
    selector_entries = [
        entry for entry in layer_record["selector_entries"]
        if (
            str(entry["selector_family"]) == "direct_set_predictor"
            and expert_id in set(map(int, entry["experts"]))
        ) or (
            str(entry["selector_family"]) != "direct_set_predictor"
            and int(entry["expert_id"]) == expert_id
        )
    ]
    for entry in selector_entries:
        family = str(entry["selector_family"])
        started = time.perf_counter()
        diagnostic: dict[str, Any]
        if family == "direct_set_predictor":
            q2_features = np.stack((gate2, up2, runtime_h2), axis=1)
            predictor_output, compute = _direct_prediction(
                arrays, entry, layer_record, expert_id, runtime_activation,
                q2_features, abc_features,
            )
            candidate_logits = predictor_output.candidate_logits[0].detach().cpu().numpy()
            apply_logits = predictor_output.apply_logits[0].detach().cpu().numpy()
            candidate_order = np.lexsort((np.arange(outputs.units), -candidate_logits))
            apply_order = np.lexsort((np.arange(outputs.units), -apply_logits))
            direct = np.asarray(apply_order[: int(config["applied_units"])], np.int64)
            candidate_list = list(map(int, direct))
            candidate_set = set(candidate_list)
            for unit in candidate_order:
                if int(unit) not in candidate_set:
                    candidate_list.append(int(unit))
                    candidate_set.add(int(unit))
                    if len(candidate_list) == int(config["candidate_units"]):
                        break
            candidates = np.asarray(candidate_list, np.int64)
            expert_metadata_bytes = int(compute["expert_metadata_bytes"])
            layer_shared_bytes = int(compute["layer_shared_bytes"])
            local_abc_bytes = int(compute["abc_bytes"])
            diagnostic = {
                "delta_gate_relative_mse": float("nan"),
                "delta_up_relative_mse": float("nan"),
                "hidden_q4_relative_mse": float("nan"),
                "training_cohort": str(entry["training_cohort"]),
                "feature_variant": str(entry["feature_variant"]),
                "objective_variant": str(entry["objective_variant"]),
                "set_loss_enabled": bool(entry["set_loss_enabled"]),
                "model_serialization": str(entry["model_serialization"]),
                "feature_transform": str(entry["feature_transform"]),
                "teacher_marginal_transform": str(entry["teacher_marginal_transform"]),
                "validation_early_stopping": False,
                "head_mode": str(entry["head_mode"]),
                "candidate_head_supervision": str(entry["candidate_coverage_semantics"]),
                "candidate_coverage_weight": float(entry["candidate_coverage_weight"]),
                "candidate_set_loss_enabled": False,
                "static_unit_bias_semantics": str(entry["static_unit_bias_semantics"]),
                "runtime_feature_precision": str(entry["runtime_feature_precision"]),
                "pq_runtime_response_precision": str(entry["pq_runtime_response_precision"]),
            }
            selection_regime = "deployable_h0_direct_set_predictor"
        else:
            predicted_gate, predicted_up, compute = _selector_prediction(
                arrays, entry, runtime_activation,
            )
            predicted_h4 = silu(gate2 + predicted_gate) * (up2 + predicted_up)
            score = complete_unit_scores(runtime_h2, predicted_h4, encoded_metadata)
            order = descending_order(score)
            candidates = np.asarray(order[: int(config["candidate_units"])], np.int64)
            direct = np.asarray(order[: int(config["applied_units"])], np.int64)
            expert_metadata_bytes = int(entry["expert_specific_bytes"])
            layer_shared_bytes = int(entry["layer_shared_bytes"])
            local_abc_bytes = int(abc_bytes)
            diagnostic = {
                "delta_gate_relative_mse": float(np.square(predicted_gate - exact_delta_gate).sum()) / max(float(np.square(exact_delta_gate).sum()), 1e-30),
                "delta_up_relative_mse": float(np.square(predicted_up - exact_delta_up).sum()) / max(float(np.square(exact_delta_up).sum()), 1e-30),
                "hidden_q4_relative_mse": float(np.square(predicted_h4 - exact_runtime_h4).sum()) / max(float(np.square(exact_runtime_h4).sum()), 1e-30),
                "fit_objective": str(entry.get("fit_objective", entry.get("pq_fit_objective", ""))),
                "quantization_training": str(entry.get("quantization_training", "")),
                "block_scale_provenance": str(entry.get("block_scale_provenance", "none")),
                "runtime_implementation": str(compute.get("runtime_implementation", "")),
                "analytical_contract": str(compute.get("analytical_contract", "")),
                "promotable": bool(entry.get("promotable", True)),
            }
            selection_regime = "deployable_h0_synopsis_direct_score"
        compute = _charge_selector_score_payload(
            compute, units=outputs.units, abc_metadata_bytes=local_abc_bytes,
            abc_score_required=family != "direct_set_predictor",
        )
        diagnostic = {
            **diagnostic,
            "selector_activation_payload_already_read": bool(compute["activation_payload_already_read"]),
            "selector_activation_payload_bytes_read": int(compute["activation_payload_bytes_read"]),
            "selector_q2_unit_feature_bytes_read": int(compute["q2_unit_feature_bytes_read"]),
            "selector_abc_metadata_bytes_read": int(compute["abc_metadata_bytes_read"]),
        }
        runtime = time.perf_counter() - started
        direct_mask = np.zeros(outputs.units, bool)
        direct_mask[direct] = True
        direct_accounting = accounting_fields(
            config,
            expert_metadata_bytes=expert_metadata_bytes,
            layer_shared_bytes=layer_shared_bytes,
            abc_bytes=local_abc_bytes,
            compute_macs=int(compute["compute_macs"]),
            abc_score_units=int(compute["abc_score_units"]),
            abc_score_compute_macs=int(compute["abc_score_compute_macs"]),
            compute_additions=int(compute["compute_additions"]),
            scale_multiplications=int(compute.get("scale_multiplications", 0)),
            selector_bytes_read=int(compute["bytes_read"]),
            fetched_units=int(config["applied_units"]),
            applied_units=int(config["applied_units"]),
        )
        base_damage = _require_positive_finite(
            float(np.ones(outputs.units) @ gram @ np.ones(outputs.units)),
            "selector base damage",
        )
        result["selector_frontier"].append({
            **metadata,
            "selector_config_id": str(entry["config_id"]),
            "selector_family": family,
            "selection_regime": selection_regime,
            "applied_units": int(config["applied_units"]),
            "recovery": 1.0 - set_damage(gram, direct_mask) / base_damage,
            "selector_runtime_ms": 1000.0 * runtime,
            **diagnostic,
            **direct_accounting,
        })
        candidate_accounting = accounting_fields(
            config,
            expert_metadata_bytes=expert_metadata_bytes,
            layer_shared_bytes=layer_shared_bytes,
            abc_bytes=local_abc_bytes,
            compute_macs=int(compute["compute_macs"]),
            abc_score_units=int(compute["abc_score_units"]),
            abc_score_compute_macs=int(compute["abc_score_compute_macs"]),
            compute_additions=int(compute["compute_additions"]),
            scale_multiplications=int(compute.get("scale_multiplications", 0)),
            selector_bytes_read=int(
                compute["bytes_read"] + int(config["candidate_units"]) * UNIT_PACKET_BYTES
            ),
            fetched_units=int(config["candidate_units"]),
            applied_units=int(config["applied_units"]),
        )
        for row in candidate_rerank_records(
            gram, candidates, int(config["applied_units"]),
            independent_scores=independent_abc_score,
            predicted_apply=direct,
            activation_payload_already_read=bool(compute["activation_payload_already_read"]),
            q2_payload_already_read=bool(compute["q2_payload_already_read"]),
            abc_payload_already_read=bool(compute["abc_payload_already_read"]),
            abc_metadata_bytes=int(local_abc_bytes),
            proxy_rank=metric_proxy_rank,
            q2_gate_up_scale_payload_already_read=bool(
                compute["q2_gate_up_scale_payload_already_read"]
            ),
        ):
            result["candidate_frontier"].append({
                **metadata,
                **row,
                "selector_config_id": str(entry["config_id"]),
                "selector_family": family,
                "candidate_units": int(config["candidate_units"]),
                "candidate_cap_units": int(config["candidate_units"]),
                "candidate_cap_pass": True,
                "applied_units": int(config["applied_units"]),
                "selector_runtime_ms": 1000.0 * runtime,
                **diagnostic,
                **candidate_accounting,
                "total_selector_rerank_compute_macs": int(
                    candidate_accounting["selector_compute_macs"]
                    + row["rerank_incremental_compute_macs"]
                ),
                "total_selector_rerank_bytes_read": int(
                    total_candidate_bytes_read(
                        candidate_accounting["selector_bytes_read"],
                        row["rerank_incremental_bytes_read"],
                        int(config["candidate_units"]),
                    )
                ),
                "total_selector_compute_macs": int(
                    candidate_accounting["selector_compute_macs"]
                    + row["rerank_incremental_compute_macs"]
                ),
                "total_selector_bytes_read": int(
                    total_candidate_bytes_read(
                        candidate_accounting["selector_bytes_read"],
                        row["rerank_incremental_bytes_read"],
                        int(config["candidate_units"]),
                    )
                ),
            })

    return result


def load_checkpoint_state(
    output: Path, expected: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[dict[str, Any]]]:
    path = output / "run_facts.json"
    if not path.exists():
        return {name: [] for name in ARTIFACTS}, [], []
    facts = json.loads(path.read_text())
    for key, value in expected.items():
        if facts.get(key) != value:
            raise RuntimeError(f"resume provenance mismatch for {key}: {facts.get(key)!r} != {value!r}")
    if facts.get("completed") is True:
        raise RuntimeError("validation output is already complete")
    completed = list(map(str, facts.get("completed_work_units", [])))
    allowed = set(completed)
    state: dict[str, list[dict[str, Any]]] = {}
    for name, filename in ARTIFACTS.items():
        artifact = output / filename
        rows = pd.read_parquet(artifact).to_dict("records") if artifact.exists() else []
        state[name] = [
            row for row in rows
            if f"validation:{int(row['layer'])}:{int(row['expert_id'])}" in allowed
        ]
    history = list(facts.get("failure_history", [])) + list(facts.get("failures", []))
    return state, completed, history


def validate_phase(args: argparse.Namespace, config: dict[str, Any]) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    validation_device = torch.device(args.device)
    hardware = runtime_provenance(validation_device)
    audit, hashes = base.verify_locked_inputs(
        config, args.config, args.exact_captures, args.checkpoint, args.trees, "exact_checkpoint",
    )
    data = load_capture_admitted(args.exact_captures, ["train", "validation"])
    if "test" in set(map(str, np.unique(data["split"]))):
        raise RuntimeError("test row entered validation process")
    fit_facts_path = args.fit_dir / "fit_facts.json"
    manifest_path = args.fit_dir / "set_utility_fit_manifest.json"
    fit_facts = json.loads(fit_facts_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if not fit_facts.get("completed") or fit_facts.get("validation_or_test_rows_used") is not False:
        raise RuntimeError("fit facts do not prove a completed training-only fit")
    if fit_facts.get("test_rows_admitted") is not False or manifest.get("test_rows_admitted") is not False:
        raise RuntimeError("fit bundle does not prove test rows remained unadmitted")
    if fit_facts.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("fit manifest changed")
    if manifest.get("config_sha256") != sha256(args.config) or manifest.get("fit_split") != "train":
        raise RuntimeError("fit manifest provenance changed")
    if (
        fit_facts.get("direct_candidate_head_supervision")
        != JOINT_CANDIDATE_COVERAGE_SEMANTICS
        or manifest.get("direct_candidate_head_supervision")
        != JOINT_CANDIDATE_COVERAGE_SEMANTICS
    ):
        raise RuntimeError("fit bundle candidate-head physical semantics changed")
    require_runtime_provenance_match(
        fit_facts.get("runtime_provenance"), hardware, "fit facts to validation",
    )
    require_runtime_provenance_match(
        manifest.get("runtime_provenance"), hardware, "fit manifest to validation",
    )
    for layer in map(int, config["layers"]):
        require_runtime_provenance_match(
            manifest.get("layers", {}).get(str(layer), {}).get("runtime_provenance"),
            hardware,
            f"fit layer {layer} manifest record to validation",
        )
    request_audit = require_locked_request_audit(fit_facts, manifest, config)
    validation_plan = evaluation_plan(data, config)
    validation_scope_audit = audit_validation_plan(validation_plan, config)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {projection: tree_from_record(tree_records[projection]) for projection in PROJECTIONS}
    facts = {
        "completed": False, "phase": "validate", "failures": [],
        "run_id": config["run_id"], "evaluation_split": "validation",
        "config_sha256": sha256(args.config), "capture_sha256": hashes["capture_sha256"],
        "tree_sha256": hashes["tree_sha256"],
        "checkpoint_config_sha256": hashes["config_sha256"],
        "checkpoint_index_sha256": hashes["index_sha256"],
        "checkpoint_audit": audit, "fit_manifest_sha256": sha256(manifest_path),
        "fit_facts_sha256": sha256(fit_facts_path), "request_separation_verified": True,
        "cross_capture_request_id_audit": request_audit,
        "direct_candidate_head_supervision": JOINT_CANDIDATE_COVERAGE_SEMANTICS,
        "runtime_provenance": hardware,
        "validation_scope_audit": validation_scope_audit,
        "expected_unique_invocations": int(config["expected_validation_unique_invocations"]),
        "expected_layer_expert_cells": int(config["expected_validation_layer_expert_cells"]),
        "test_rows_admitted": False, "test_rows_used": False,
        "monolithic_source_contains_test_rows": True, "codec_locked": True,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "teacher_core_sha256": sha256(EXPERIMENT / "src/oracle_study/unit_set_teacher.py"),
        "oracle_core_sha256": sha256(EXPERIMENT / "src/oracle_study/set_utility_oracles.py"),
        "selector_core_sha256": sha256(EXPERIMENT / "src/oracle_study/set_utility_selector.py"),
        "direct_predictor_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/direct_set_predictor.py"
        ),
        "neuron_selector_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/neuron_selector.py"
        ),
        "prior_runner_sha256": sha256(
            EXPERIMENT / "scripts/run_neuron_selector_distillation.py"
        ),
        "base_runner_sha256": sha256(
            EXPERIMENT / "scripts/run_sparse_streaming_study.py"
        ),
        "sparse_streaming_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/sparse_streaming.py"
        ),
        "mxfp4_embed_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/mxfp4_embed.py"
        ),
        "selective_pages_runner_sha256": sha256(
            EXPERIMENT / "scripts/run_mxfp4_selective_pages.py"
        ),
        "phase_a_runner_sha256": sha256(
            EXPERIMENT / "scripts/run_phase_a_remote.py"
        ),
        "device": hardware["device"],
        "gpu": hardware["gpu_names_used"][0] if hardware["gpu_names_used"] else None,
        "host": hardware["host"],
        "expert_static_precompute": {
            "down_metric_gram_once_per_layer_expert_proxy": True,
            "encoded_fp16_abc_once_per_layer_expert_proxy": True,
        },
        "hybrid_oracle_scope": {
            "max_invocations_per_expert": int(config["max_hybrid_oracle_invocations_per_expert"]),
            "selection": "every admitted exact-validation routed occurrence",
            "backend": "torch GPU primitive base-Gram plus deterministic CPU correlation update",
        },
        "selector_runtime_activation_precision": "FP16-rounded values evaluated in FP32 reference kernels",
        "gram_output_qenergy_parity": "asserted for base and exact-selected residual on every validation invocation",
    }
    for dependency in (
        "runner_sha256", "teacher_core_sha256", "oracle_core_sha256",
        "selector_core_sha256", "direct_predictor_core_sha256",
        "neuron_selector_core_sha256", "prior_runner_sha256", "base_runner_sha256",
        "sparse_streaming_core_sha256", "mxfp4_embed_core_sha256",
        "selective_pages_runner_sha256", "phase_a_runner_sha256",
    ):
        if fit_facts.get(dependency) != facts[dependency]:
            raise RuntimeError(
                f"fit/validation dependency mismatch for {dependency}: "
                f"{fit_facts.get(dependency)!r} != {facts[dependency]!r}"
            )
    resume = {key: facts[key] for key in (
        "run_id", "evaluation_split", "config_sha256", "capture_sha256", "tree_sha256",
        "checkpoint_config_sha256", "checkpoint_index_sha256", "fit_manifest_sha256",
        "fit_facts_sha256", "runner_sha256", "teacher_core_sha256", "oracle_core_sha256",
        "selector_core_sha256", "direct_predictor_core_sha256",
        "neuron_selector_core_sha256", "prior_runner_sha256", "base_runner_sha256",
        "sparse_streaming_core_sha256", "mxfp4_embed_core_sha256",
        "selective_pages_runner_sha256", "phase_a_runner_sha256",
        "cross_capture_request_id_audit", "validation_scope_audit",
        "runtime_provenance",
    )}
    state, completed, history = load_checkpoint_state(args.output, resume)
    facts["completed_work_units"] = completed
    facts["failure_history"] = history
    facts_path = args.output / "run_facts.json"
    atomic_json(facts_path, facts)
    try:
        loaded_layer = None
        arrays: dict[str, np.ndarray] = {}
        layer_record: dict[str, Any] = {}
        for layer, expert, stratum, records, ranks in validation_plan:
            work_unit = f"validation:{layer}:{expert}"
            if work_unit in set(completed):
                continue
            if loaded_layer != layer:
                arrays, layer_record = load_fit_layer(args.fit_dir, manifest, layer)
                loaded_layer = layer
            local = layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
            )
            matrices = decode_expert(args.checkpoint, index, trees, layer, expert)
            q2 = tuple(matrices[name][0] for name in PROJECTIONS)
            q4 = tuple(matrices[name][2] for name in PROJECTIONS)
            static_down_metric = down_metric_gram(
                q2[2], q4[2], proxy=proxy, beta=float(proxy_facts["beta"]),
            )
            exact_abc = unit_score_metadata(
                q2[2], q4[2], proxy=proxy, beta=float(proxy_facts["beta"]),
            )
            encoded_abc, abc_bytes = encode_unit_score_metadata(exact_abc, "fp16")
            hybrid_limit = int(config["max_hybrid_oracle_invocations_per_expert"])
            for invocation_index, (record, router_rank) in enumerate(zip(records, ranks)):
                metadata = base.common_metadata(
                    "exact_checkpoint", "validation", local, int(record), int(router_rank),
                    layer, expert, stratum,
                )
                rows = evaluate_invocation(
                    matrices, np.asarray(local["x"][record], np.float32), proxy,
                    float(proxy_facts["beta"]), static_down_metric, encoded_abc,
                    int(abc_bytes), metadata, arrays, layer_record, config,
                    run_hybrid_oracle=invocation_index < hybrid_limit,
                    hybrid_device=args.device,
                )
                for name in ARTIFACTS:
                    state[name].extend(rows[name])
            completed.append(work_unit)
            for name, filename in ARTIFACTS.items():
                atomic_parquet(args.output / filename, state[name])
            facts["completed_work_units"] = completed
            facts["artifact_row_counts"] = {name: len(value) for name, value in state.items()}
            atomic_json(facts_path, facts)
            del matrices
            gc.collect()
        identities = {
            tuple(row[column] for column in IDENTITY)
            for rows in state.values() for row in rows
        }
        expected = int(config["expected_validation_unique_invocations"])
        if len(identities) != expected:
            raise RuntimeError(f"observed {len(identities)} validation invocations != expected {expected}")
        observed_cells = {
            (int(identity[4]), int(identity[5])) for identity in identities
        }
        locked_cells = set(locked_validation_cells(config))
        if observed_cells != locked_cells:
            raise RuntimeError(
                f"artifact layer/expert cells {sorted(observed_cells)!r} "
                f"!= locked {sorted(locked_cells)!r}"
            )
        expected_hybrid_rows = (
            expected * len(config["hybrid_budgets_bpw"]) * 5
        )
        if len(state["hybrid_oracle_frontier"]) != expected_hybrid_rows:
            raise RuntimeError(
                f"hybrid broad coverage {len(state['hybrid_oracle_frontier'])} "
                f"!= expected {expected_hybrid_rows}"
            )
        accounting_rows = []
        selector = pd.DataFrame(state["selector_frontier"])
        candidate = pd.DataFrame(state["candidate_frontier"])
        template = pd.DataFrame(state["template_frontier"])
        if not selector.empty:
            columns = [
                "selector_config_id", "selector_family", "expert_specific_metadata_bytes",
                "selector_abc_score_units", "selector_abc_score_compute_macs",
                "selector_abc_score_macs_per_unit",
                "selector_abc_score_mac_convention",
                "layer_shared_metadata_bytes", "layer_shared_amortized_bytes_per_expert",
                "abc_metadata_bytes", "selector_metadata_bytes_per_expert", "selector_metadata_bpw",
                "selector_compute_macs", "selector_compute_additions",
                "selector_scale_multiplications", "selector_bytes_read",
                "selector_activation_payload_already_read",
                "selector_activation_payload_bytes_read",
                "selector_q2_unit_feature_bytes_read",
                "selector_abc_metadata_bytes_read", "storage_multiplier",
            ]
            accounting_rows = selector[columns].drop_duplicates().to_dict("records")
        template_accounting_rows = []
        if not template.empty:
            template_columns = [
                "template_cohort", "template_count_effective",
                "template_bank_metadata_bytes", "template_classifier_metadata_bytes",
                "selector_metadata_bytes_per_expert", "selector_metadata_bpw",
                "storage_multiplier", "selector_metadata_semantics",
            ]
            template_accounting_rows = (
                template[template_columns].drop_duplicates().to_dict("records")
            )
        rerank_accounting_rows = []
        if not candidate.empty:
            rerank_columns = [
                "selector_config_id", "selector_family", "rerank_semantics",
                "rerank_role", "candidate_units", "rerank_incremental_compute_macs",
                "rerank_incremental_bytes_read",
                "activation_payload_already_read", "q2_payload_already_read",
                "abc_payload_already_read", "rerank_activation_payload_bytes_read",
                "rerank_q2_unit_feature_bytes_read", "rerank_abc_metadata_bytes_read",
                "rerank_q2_gate_up_parent_payload_required",
                "rerank_q2_gate_up_parent_payload_accounted",
                "rerank_q2_gate_up_parent_code_bytes_read",
                "rerank_q2_gate_up_scale_bytes_read",
                "rerank_q2_gate_up_scale_payload_already_read",
                "rerank_q4_response_scale_multiplications",
                "selector_abc_score_units", "selector_abc_score_compute_macs",
                "selector_abc_score_macs_per_unit",
                "selector_abc_score_mac_convention",
                "rerank_candidate_packet_bytes_read", "rerank_candidate_packet_contents",
                "rerank_resident_q2_down_payload_required",
                "rerank_resident_q2_down_payload_accounted",
                "rerank_packet_bytes_already_in_selector_bytes_read",
                "rerank_q4_response_units", "rerank_q4_response_compute_macs",
                "rerank_down_correction_compute_macs",
                "rerank_correction_workspace_bytes", "rerank_interaction_gram_units",
                "rerank_interaction_gram_euclidean_compute_macs",
                "rerank_interaction_proxy_rank",
                "rerank_interaction_proxy_projection_compute_macs",
                "rerank_interaction_proxy_gram_compute_macs",
                "rerank_interaction_proxy_scale_multiplications",
                "rerank_proxy_metadata_bytes_read",
                "rerank_interaction_gram_compute_macs",
                "rerank_abc_score_macs_per_unit", "rerank_abc_score_mac_convention",
                "rerank_non_mac_operations",
                "rerank_accounting_is_lower_bound", "rerank_accounting_bound",
                "rerank_unaccounted_overhead",
                "total_selector_rerank_compute_macs",
                "total_selector_rerank_bytes_read",
                "total_selector_compute_macs", "total_selector_bytes_read",
            ]
            rerank_accounting_rows = (
                candidate[rerank_columns].drop_duplicates().to_dict("records")
            )
        all_storage = [
            *(float(row["storage_multiplier"]) for row in accounting_rows),
            *(float(row["storage_multiplier"]) for row in template_accounting_rows),
        ]
        atomic_json(args.output / "selector_compute_storage_accounting.json", {
            "schema_version": 4,
            "physical_interface": {
                "candidate_units": 256, "applied_units": 192,
                "pages": 768, "bytes": 393216,
            },
            "rows": accounting_rows,
            "q4_response_resident_payload_contract": {
                "q2_parent_code_bytes_per_unit": Q2_GATE_UP_PARENT_CODE_BYTES_PER_UNIT,
                "q2_scale_bytes_per_unit": Q2_GATE_UP_SCALE_BYTES_PER_UNIT,
                "scale_multiplications_per_unit": Q4_RESPONSE_SCALE_MULTIPLICATIONS_PER_UNIT,
                "candidate_packet_bytes_per_unit": UNIT_PACKET_BYTES,
                "candidate_packet_is_suffix_only": True,
                "scale_bytes_deduplicated_when_selector_already_read": True,
            },
            "selector_rows": accounting_rows,
            "template_bank_rows": template_accounting_rows,
            "candidate_rerank_rows": rerank_accounting_rows,
            "all_storage_multipliers_below_5x": all(value < 5.0 for value in all_storage),
            "candidate_rerank_semantics": sorted(set(
                candidate.get("rerank_semantics", pd.Series(dtype=str)).astype(str)
            )),
            "promotion_primary_rerank": "exact_independent_abc_within_fetched_candidates",
            "contained_interaction_rerank_is_systems_oracle": True,
        })
        facts["expected_unique_invocations"] = expected
        facts["observed_unique_invocations"] = len(identities)
        facts["observed_layer_expert_cells"] = len(observed_cells)
        facts["expected_hybrid_rows"] = expected_hybrid_rows
        facts["observed_hybrid_rows"] = len(state["hybrid_oracle_frontier"])
        facts["completed"] = True
        facts["completed_unix"] = time.time()
        facts["accounting_sha256"] = sha256(args.output / "selector_compute_storage_accounting.json")
        atomic_json(facts_path, facts)
    except Exception as error:
        facts["failures"].append({
            "type": type(error).__name__, "message": str(error), "unix": time.time(),
        })
        atomic_json(facts_path, facts)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit", "validate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--cross-captures", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    validate_study_contract(config)
    if args.phase == "fit":
        if args.cross_captures is None:
            raise RuntimeError("fit requires --cross-captures; both locked training sources are mandatory")
        fit_phase(args, config)
    else:
        if args.fit_dir is None:
            raise RuntimeError("validate requires --fit-dir")
        if args.cross_captures is not None:
            raise RuntimeError("validation cannot load cross-reference rows")
        validate_phase(args, config)


if __name__ == "__main__":
    main()
