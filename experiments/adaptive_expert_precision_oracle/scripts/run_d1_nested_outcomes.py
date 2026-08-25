#!/usr/bin/env python3
"""Exact cached-decode downstream outcomes for sealed nested allocations.

This is the separate outcome phase of nested D1 Experiment A.  The runner has
no allocation or calibration interface: it authenticates a sealed allocation
manifest, reconstructs each complete selected state on the live current-token
activation, and executes the downstream tail from a private clone of the full
native hybrid prefix cache.  Terminal outcomes never feed back into selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.causal_replay import projection_responses  # noqa: E402
from oracle_study.d1_decode import cache_state_metrics, clone_decode_cache  # noqa: E402
from oracle_study.d1_nested_artifacts import canonical_sha256  # noqa: E402
from oracle_study.d1_nested_outcomes import (  # noqa: E402
    ALLOCATION_CONFIG_FILE_SHA256,
    AUTHENTICATED_CAPTURE_CONFIG_PATH,
    AUTHENTICATED_CAPTURE_CONFIG_SHA256,
    CACHE_FILE,
    CONTRAST_FILE,
    OUTCOME_COMPATIBILITY_SCHEMA,
    OUTCOME_SCHEMA,
    PROPAGATION_FILE,
    QUALITY_FILE,
    ROUTE_MODES,
    RUN_FACTS_FILE,
    ZERO_FILE,
    OutcomeAllocation,
    OutcomeRequest,
    file_sha256,
    join_request_manifest,
    load_outcome_plan,
    outcome_input_facts,
    physical_execution_identity,
    require_zero_dose_parity,
    route_mode_contrasts,
    validate_complete_allocation_grid,
    validate_outcome_config,
    validate_outcome_compatibility_protocol,
    validate_plan_against_config,
)
from oracle_study.split_interaction_field import split_state_output  # noqa: E402
from capture_d1_nested_exact_decode import (  # noqa: E402
    _gpu_gate,
    validate_capture_inputs,
    validate_checkpoint_pins,
)
from run_d1_downstream_tail_kl import (  # noqa: E402
    CachedDecodeObserver,
    DecodeObservation,
    _np,
    _observation_maxima,
    _run_candidate,
    _zero_metrics,
)
from oracle_study.d1_decode_tail import (  # noqa: E402
    classified_cache_metrics,
    current_token_quality_metrics,
    downstream_route_rows,
)
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
from run_same_host_causal_controls import (  # noqa: E402
    _load_model,
    atomic_json,
    atomic_parquet,
)
import run_average_rate_allocation as pr13  # noqa: E402
import run_set_utility_distillation as set_study  # noqa: E402


CELL_FACTS = "d1_nested_outcome_cell_facts.json"
HOST_FACTS = "d1_nested_outcome_host_facts.json"
Q4_ROUTED_MAX_ABS_ATOL = 0.125
FROZEN_ALLOCATION_CONFIG_PATH = (
    EXPERIMENT
    / "configs/qwen36_mxfp4_d1_nested_safe_allocation_20260825_v2.json"
)
OUTCOME_CODE_BUNDLE_SCHEMA = "pr13_d1_nested_outcome_code_bundle_v1"
OUTCOME_CODE_BUNDLE_ROOTS = ("scripts", "src/oracle_study")
OUTCOME_CODE_REQUIRED_MEMBERS = {
    "scripts/run_d1_nested_outcomes.py",
    "scripts/run_d1_downstream_tail_kl.py",
    "src/oracle_study/d1_nested_outcomes.py",
    "src/oracle_study/d1_decode_tail.py",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _cell_paths(output: Path, split: str, layer: int) -> dict[str, Path]:
    root = Path(output) / "cells" / str(split) / f"layer_{int(layer):02d}"
    return {
        "root": root,
        "quality": root / QUALITY_FILE,
        "propagation": root / PROPAGATION_FILE,
        "cache": root / CACHE_FILE,
        "zero": root / ZERO_FILE,
        "facts": root / CELL_FACTS,
    }


def _outcome_code_bundle() -> dict[str, Any]:
    """Hash the complete experiment Python implementation deterministically."""

    members = []
    for relative_root in OUTCOME_CODE_BUNDLE_ROOTS:
        root = EXPERIMENT / relative_root
        for path in sorted(root.rglob("*.py")):
            if not path.is_file():
                continue
            members.append({
                "path": path.relative_to(EXPERIMENT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    observed = {str(member["path"]) for member in members}
    missing = sorted(OUTCOME_CODE_REQUIRED_MEMBERS.difference(observed))
    if missing:
        raise RuntimeError(f"outcome code bundle is incomplete: {missing}")
    inventory = {
        str(member["path"]): {
            "sha256": str(member["sha256"]),
            "bytes": int(member["bytes"]),
        }
        for member in members
    }
    candidate_identity = {
        "schema": "pr13_d1_nested_code_bundle_v1",
        "files": len(inventory),
        "canonical_sha256": canonical_sha256(inventory),
    }
    payload = {
        "schema": OUTCOME_CODE_BUNDLE_SCHEMA,
        "scope": "all_python_under_experiment_scripts_and_src_oracle_study",
        "roots": list(OUTCOME_CODE_BUNDLE_ROOTS),
        "files": len(members),
        "members": members,
        "allocation_candidate_code_identity": candidate_identity,
    }
    return {
        **payload,
        "sha256": canonical_sha256(payload),
    }


def _authenticate_outcome_code_bundle(
    plan: Any,
    protocol: Mapping[str, Any],
    *,
    allocation_config_sha256: str,
    capture_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate separately versioned outcome code against a checked-in seal."""

    bundle = _outcome_code_bundle()
    observed = bundle["allocation_candidate_code_identity"]
    expected = dict(plan.calibration_candidate_code_identity)
    compatibility = validate_outcome_compatibility_protocol(
        protocol,
        allocation_config_file_sha256=allocation_config_sha256,
        capture_config_file_sha256=AUTHENTICATED_CAPTURE_CONFIG_SHA256,
        allocation_candidate_code_identity=expected,
        outcome_code_identity=observed,
        allocation_hardware_execution_path=load_json(
            FROZEN_ALLOCATION_CONFIG_PATH
        )["hardware_execution_path"],
        capture_hardware_execution_path=capture_config[
            "hardware_execution_path"
        ],
    )
    return {
        **bundle,
        "allocation_code_identity_equal_to_outcome_code_identity": False,
        "compatibility_protocol": compatibility,
    }


def _validate_execution_stack(
    config: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact GPU/Torch/CUDA route-label execution stack."""

    expected = config.get("authenticated_capture_execution_stack")
    if not isinstance(expected, Mapping):
        raise RuntimeError("authenticated capture execution stack is absent")
    observed = {
        "gpu": str(hardware.get("gpu", "")),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    mismatches = {
        field: {"expected": str(expected.get(field)), "observed": value}
        for field, value in observed.items()
        if value != str(expected.get(field))
    }
    if mismatches:
        raise RuntimeError(
            f"authenticated capture execution stack mismatch: {mismatches}"
        )
    return dict(expected)


def _input_pins(
    *,
    allocation_config_sha256: str,
    capture_hashes: Mapping[str, str],
    checkpoint_hashes: Mapping[str, str],
    selected_tree_sha256: str,
    allocation_facts: Mapping[str, Any],
    execution_stack: Mapping[str, Any],
    outcome_code_bundle: Mapping[str, Any],
    q4_routed_max_abs_atol: float,
) -> dict[str, Any]:
    return {
        "allocation_config_sha256": str(allocation_config_sha256),
        "authenticated_capture_config_sha256": str(
            capture_hashes["config_sha256"]
        ),
        "request_manifest_sha256": str(capture_hashes["manifest_sha256"]),
        "request_manifest_facts_sha256": str(
            capture_hashes["manifest_facts_sha256"]
        ),
        "checkpoint_config_sha256": str(
            checkpoint_hashes["checkpoint_config_sha256"]
        ),
        "checkpoint_index_sha256": str(
            checkpoint_hashes["checkpoint_index_sha256"]
        ),
        "selected_tree_sha256": str(selected_tree_sha256),
        "authenticated_capture_execution_stack": dict(execution_stack),
        "outcome_code_bundle": dict(outcome_code_bundle),
        "q4_routed_max_abs_atol": float(q4_routed_max_abs_atol),
        **dict(allocation_facts),
    }


def _completed_cell(
    paths: Mapping[str, Path],
    *,
    split: str,
    layer: int,
    pins: Mapping[str, Any],
    allocations: Sequence[OutcomeAllocation],
) -> bool:
    facts_path = paths["facts"]
    if not facts_path.is_file():
        return False
    facts = load_json(facts_path)
    if (
        facts.get("schema") != OUTCOME_SCHEMA
        or facts.get("completed") is not True
        or facts.get("split") != split
        or int(facts.get("injection_layer", -1)) != int(layer)
        or int(facts.get("allocations", -1)) != len(allocations)
        or facts.get("input_pins") != dict(pins)
        or facts.get("route_modes") != list(ROUTE_MODES)
        or facts.get("one_exact_pretoken_cache_per_request_across_all_layers") is not True
        or facts.get("one_paired_native_baseline_per_request_across_all_layers") is not True
        or facts.get("logical_rows_fanned_out_from_physical_execution") is not True
        or facts.get("every_zero_route_mode_physically_executed") is not True
    ):
        raise RuntimeError("existing outcome cell facts cannot authenticate resume")
    files = facts.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("existing outcome cell file inventory is absent")
    expected_rows = {
        "quality": len(allocations) * len(ROUTE_MODES),
        "propagation": len(allocations) * len(ROUTE_MODES) * (39 - int(layer)),
        "cache": len(allocations) * len(ROUTE_MODES),
        "zero": len({row.request_id for row in allocations}) * len(ROUTE_MODES),
    }
    frames: dict[str, pd.DataFrame] = {}
    for key in ("quality", "propagation", "cache", "zero"):
        path = paths[key]
        record = files.get(path.name)
        if (
            not isinstance(record, Mapping)
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
            or int(record.get("rows", -1)) != expected_rows[key]
        ):
            raise RuntimeError(f"existing outcome cell artifact changed: {path}")
        frames[key] = pd.read_parquet(path)
        if len(frames[key]) != expected_rows[key]:
            raise RuntimeError(f"existing outcome cell row count changed: {path}")
    _validate_cell_tables(
        frames, split=split, layer=layer, allocations=allocations,
    )
    physical_states = frames["quality"][[
        "request_id", "physical_execution_id",
    ]].drop_duplicates()
    physical_executions = frames["quality"][[
        "request_id", "physical_execution_id", "route_mode",
    ]].drop_duplicates()
    if (
        int(facts.get("requests", -1))
        != len({row.request_id for row in allocations})
        or int(facts.get("logical_candidate_comparisons", -1))
        != expected_rows["quality"]
        or int(facts.get("zero_candidate_executions", -1))
        != expected_rows["zero"]
        or int(facts.get("physical_allocation_states", -1))
        != len(physical_states)
        or int(facts.get("physical_candidate_executions", -1))
        != len(physical_executions)
    ):
        raise RuntimeError("existing outcome cell execution provenance changed")
    return True


def _validate_cell_tables(
    frames: Mapping[str, pd.DataFrame],
    *,
    split: str,
    layer: int,
    allocations: Sequence[OutcomeAllocation],
) -> None:
    candidate_columns = [
        "arm", "rate_pages_per_expert", "request_id", "selected_state_sha256",
        "route_mode",
    ]
    expected_candidates = {
        (row.arm, row.rate, row.request_id, row.state_sha256, route_mode)
        for row in allocations
        for route_mode in ROUTE_MODES
    }
    provenance_columns = [
        "zero_dose_gate_id", "physical_execution_id",
        "physical_expert_state_sha256", "physical_delta_sha256",
        "physical_execution_logical_fanout",
        "physical_execution_reuse_ordinal", "physical_execution_reused",
        "physical_representative_arm",
        "physical_representative_rate_pages_per_expert",
        "physical_representative_selected_state_sha256",
    ]
    for name in ("quality", "cache"):
        frame = frames[name]
        required = set(
            candidate_columns + provenance_columns
            + ["schema", "split", "injection_layer"]
        )
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{name} cell table is missing identity columns")
        if (
            set(frame["schema"].astype(str)) != {OUTCOME_SCHEMA}
            or set(frame["split"].astype(str)) != {split}
            or set(frame["injection_layer"].astype(int)) != {int(layer)}
            or frame.duplicated(candidate_columns).any()
        ):
            raise RuntimeError(f"{name} cell identity grid changed")
        actual = set(frame[candidate_columns].itertuples(index=False, name=None))
        if actual != expected_candidates:
            raise RuntimeError(f"{name} cell candidate grid changed")

    propagation = frames["propagation"]
    propagation_columns = candidate_columns + ["observation_layer"]
    if not set(
        propagation_columns + provenance_columns
        + ["schema", "split", "injection_layer"]
    ).issubset(propagation.columns) or propagation.duplicated(propagation_columns).any():
        raise RuntimeError("propagation cell identity grid changed")
    expected_propagation = {
        (*candidate, observation_layer)
        for candidate in expected_candidates
        for observation_layer in range(int(layer) + 1, 40)
    }
    if set(propagation[propagation_columns].itertuples(index=False, name=None)) != expected_propagation:
        raise RuntimeError("propagation cell candidate/layer grid changed")
    if (
        set(propagation["schema"].astype(str)) != {OUTCOME_SCHEMA}
        or set(propagation["split"].astype(str)) != {split}
        or set(propagation["injection_layer"].astype(int)) != {int(layer)}
    ):
        raise RuntimeError("propagation cell metadata changed")

    zero = frames["zero"]
    zero_columns = ["request_id", "route_mode"]
    expected_zero = {
        (row.request_id, route_mode)
        for row in allocations
        for route_mode in ROUTE_MODES
    }
    if (
        not set(
            zero_columns + [
                "schema", "split", "injection_layer", "zero_dose_gate_id",
                "logical_allocations_authenticated",
                "paired_baseline_shared_across_injection_layers",
                "zero_physical_execution_mode",
                "zero_physical_execution_was_run",
                "zero_fully_frozen_route_hook_identity_by_construction",
                "zero_frozen_set_live_weight_all_downstream_scores_bit_identical",
                "zero_frozen_set_live_weight_max_abs",
            ]
        ).issubset(zero.columns)
        or zero.duplicated(zero_columns).any()
        or zero["zero_dose_gate_id"].astype(str).duplicated().any()
        or set(zero[zero_columns].itertuples(index=False, name=None)) != expected_zero
        or set(zero["schema"].astype(str)) != {OUTCOME_SCHEMA}
        or set(zero["split"].astype(str)) != {split}
        or set(zero["injection_layer"].astype(int)) != {int(layer)}
    ):
        raise RuntimeError("zero-dose cell identity grid changed")
    if not zero["paired_baseline_shared_across_injection_layers"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).all():
        raise RuntimeError("zero-dose rows are not bound to the request-major baseline")
    for request_id, part in zero.groupby("request_id", sort=False):
        if (
            set(map(str, part["route_mode"])) != set(ROUTE_MODES)
            or set(map(str, part["zero_physical_execution_mode"]))
            != set(ROUTE_MODES)
            or any(
                str(row.route_mode) != str(row.zero_physical_execution_mode)
                for row in part.itertuples(index=False)
            )
            or not part["zero_physical_execution_was_run"].map(
                lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
            ).all()
            or not part[
                "zero_fully_frozen_route_hook_identity_by_construction"
            ].map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value)).all()
            or not part[
                "zero_frozen_set_live_weight_all_downstream_scores_bit_identical"
            ].map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value)).all()
            or not np.all(
                part["zero_frozen_set_live_weight_max_abs"].to_numpy(np.float64)
                == 0.0
            )
        ):
            raise RuntimeError(
                f"zero-dose physical route-mode proof changed for request {request_id}"
            )
    expected_logical_by_request = {
        request_id: sum(row.request_id == request_id for row in allocations)
        for request_id in {row.request_id for row in allocations}
    }
    if any(
        int(row.logical_allocations_authenticated)
        != expected_logical_by_request[str(row.request_id)]
        for row in zero.itertuples(index=False)
    ):
        raise RuntimeError("zero-dose logical allocation fanout changed")
    zero_gate = {
        (str(row.request_id), str(row.route_mode)): str(row.zero_dose_gate_id)
        for row in zero.itertuples(index=False)
    }
    for name in ("quality", "cache", "propagation"):
        frame = frames[name]
        if any(
            str(row.zero_dose_gate_id)
            != zero_gate[(str(row.request_id), str(row.route_mode))]
            for row in frame[[
                "request_id", "route_mode", "zero_dose_gate_id",
            ]].itertuples(index=False)
        ):
            raise RuntimeError(f"{name} rows changed their authenticated zero-dose gate")

    group_columns = ["request_id", "physical_execution_id", "route_mode"]
    for identity, part in frames["quality"].groupby(
        group_columns, sort=False, dropna=False,
    ):
        fanouts = set(map(int, part["physical_execution_logical_fanout"]))
        if fanouts != {len(part)}:
            raise RuntimeError(f"physical fanout count changed for {identity}")
        ordinals = set(map(int, part["physical_execution_reuse_ordinal"]))
        if ordinals != set(range(len(part))):
            raise RuntimeError(f"physical reuse ordinals changed for {identity}")
        reused = {
            int(row.physical_execution_reuse_ordinal): bool(
                row.physical_execution_reused
            )
            for row in part.itertuples(index=False)
        }
        if reused != {ordinal: ordinal > 0 for ordinal in range(len(part))}:
            raise RuntimeError(f"physical reuse flags changed for {identity}")
        invariant = [
            "physical_expert_state_sha256", "physical_delta_sha256",
            "physical_representative_arm",
            "physical_representative_rate_pages_per_expert",
            "physical_representative_selected_state_sha256",
            "delta_nll", "ppl_ratio", "logit_kl", "top1_agreement",
            "final_hidden_mse", "realized_injected_layer_output_mse",
            "first_route_membership_change_layer", "routed_repeat_max_abs",
            "physical_candidate_seconds",
        ]
        if any(part[column].nunique(dropna=False) != 1 for column in invariant):
            raise RuntimeError(f"physical fanout outcomes differ for {identity}")

    for frame_name, frame, extra_group in (
        ("cache", frames["cache"], []),
        ("propagation", frames["propagation"], ["observation_layer"]),
    ):
        result_columns = [
            column for column in frame.columns
            if column.startswith((
                "full_cache_", "attention_kv_", "deltanet_conv_",
                "deltanet_recurrent_", "post_token_layer_cache_",
            ))
            or column in {
                "hidden_mse", "distance_from_injection",
                "route_no_change_fraction", "route_order_only_change_fraction",
                "route_membership_change_fraction", "route_top1_change_fraction",
                "router_mass_churn", "router_observation_semantics",
            }
        ]
        for identity, part in frame.groupby(
            group_columns + extra_group, sort=False, dropna=False,
        ):
            if any(part[column].nunique(dropna=False) != 1 for column in result_columns):
                raise RuntimeError(f"{frame_name} physical fanout differs for {identity}")
    require_zero_dose_parity(zero)


def _encoded_request(request: OutcomeRequest, device: torch.device) -> dict[str, torch.Tensor]:
    tokens = torch.as_tensor(
        request.prompt_token_ids, device=device, dtype=torch.long,
    ).reshape(1, -1)
    return {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
    }


class MultiLayerCachedDecodeObserver:
    """Observe one native cached-decode pass for every admitted injection layer.

    Hidden/router tensors are copied only once per decoder layer.  The small
    pre-MoE input and routed-output tensors are additionally retained for each
    admitted injection layer, allowing one exact pretoken cache and one paired
    native current-token baseline to serve every layer in the request.
    """

    def __init__(self, model: Any, injection_layers: Sequence[int]) -> None:
        layers = tuple(sorted(set(map(int, injection_layers))))
        if not layers or min(layers) < 0 or max(layers) >= 40:
            raise ValueError("multi-layer observer injection layers are invalid")
        self.injection_layers = layers
        self.injection_layer = layers[0]
        self.handles: list[Any] = []
        self.reset()
        decoder_layers = model.model.language_model.layers
        for layer_id, layer in enumerate(decoder_layers):
            def layer_hook(
                _module: Any, _inputs: Any, output: Any,
                layer_id: int = layer_id,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                self.hidden[layer_id] = value.detach().cpu().clone()

            def gate_hook(
                _module: Any, _inputs: Any, output: Any,
                layer_id: int = layer_id,
            ) -> None:
                if not isinstance(output, tuple) or len(output) < 3:
                    raise RuntimeError(
                        "router must return logits, scores, and expert IDs"
                    )
                self.router_logits[layer_id] = output[0].detach().cpu().clone()
                self.router_scores[layer_id] = output[1].detach().cpu().clone()
                self.router_ids[layer_id] = output[2].detach().cpu().clone()

            self.handles.append(layer.register_forward_hook(layer_hook))
            self.handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
            if layer_id in self.injection_layers:
                def norm_hook(
                    _module: Any, _inputs: Any, output: torch.Tensor,
                    layer_id: int = layer_id,
                ) -> None:
                    self.x_by_layer[layer_id] = output.detach().cpu().clone()

                def routed_hook(
                    _module: Any, _inputs: Any, output: torch.Tensor,
                    layer_id: int = layer_id,
                ) -> None:
                    self.routed_by_layer[layer_id] = output.detach().cpu().clone()

                self.handles.append(
                    layer.post_attention_layernorm.register_forward_hook(norm_hook)
                )
                self.handles.append(
                    layer.mlp.experts.register_forward_hook(routed_hook)
                )

    def reset(self) -> None:
        self.hidden: dict[int, torch.Tensor] = {}
        self.router_logits: dict[int, torch.Tensor] = {}
        self.router_scores: dict[int, torch.Tensor] = {}
        self.router_ids: dict[int, torch.Tensor] = {}
        self.x_by_layer: dict[int, torch.Tensor] = {}
        self.routed_by_layer: dict[int, torch.Tensor] = {}

    def set_injection_layer(self, layer: int) -> None:
        layer = int(layer)
        if layer not in self.injection_layers:
            raise ValueError(f"observer does not admit injection layer {layer}")
        self.injection_layer = layer

    def run(
        self,
        model: Any,
        encoded: Mapping[str, torch.Tensor],
        position: int,
        prefix_cache: Any | None,
    ) -> DecodeObservation:
        self.reset()
        device = encoded["input_ids"].device
        position = int(position)
        kwargs: dict[str, Any] = {
            "input_ids": encoded["input_ids"][:, position : position + 1],
            "attention_mask": encoded["attention_mask"][:, : position + 1],
            "past_key_values": prefix_cache,
            "cache_position": torch.as_tensor(
                [position], device=device, dtype=torch.long,
            ),
            "use_cache": True,
            "return_dict": True,
        }
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        expected_decoder_layers = set(range(40))
        expected_injection_layers = set(self.injection_layers)
        if (
            set(self.hidden) != expected_decoder_layers
            or set(self.router_logits) != expected_decoder_layers
            or set(self.router_scores) != expected_decoder_layers
            or set(self.router_ids) != expected_decoder_layers
            or set(self.x_by_layer) != expected_injection_layers
            or set(self.routed_by_layer) != expected_injection_layers
        ):
            raise RuntimeError("multi-layer observer missed a current-token tensor")
        active = self.injection_layer
        result = DecodeObservation(
            hidden=dict(self.hidden),
            router_logits=dict(self.router_logits),
            router_scores=dict(self.router_scores),
            router_ids=dict(self.router_ids),
            x=self.x_by_layer[active],
            routed=self.routed_by_layer[active],
            logits=output.logits[0, 0].detach().cpu().clone(),
            cache=output.past_key_values,
            elapsed_seconds=elapsed,
        )
        del output
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def advance_exact_prefix(
    observer: Any,
    model: Any,
    encoded: Mapping[str, torch.Tensor],
    position: int,
) -> tuple[Any, float]:
    """Advance exactly positions ``[0, position)`` and retain only the last cache."""

    prefix = None
    elapsed = 0.0
    for token_position in range(int(position)):
        observation = observer.run(model, encoded, token_position, prefix)
        elapsed += observation.elapsed_seconds
        next_prefix = observation.cache
        del observation
        if prefix is not None and prefix is not next_prefix:
            del prefix
        prefix = next_prefix
    if prefix is None:
        raise RuntimeError("decode position must have a non-empty exact prefix")
    return prefix, elapsed


def advance_exact_prefix_native(
    model: Any,
    encoded: Mapping[str, torch.Tensor],
    position: int,
) -> tuple[Any, float]:
    """Build one exact full hybrid pretoken cache without observer copies."""

    prefix = None
    device = encoded["input_ids"].device
    started = time.perf_counter()
    for token_position in range(int(position)):
        kwargs: dict[str, Any] = {
            "input_ids": encoded["input_ids"][:, token_position : token_position + 1],
            "attention_mask": encoded["attention_mask"][:, : token_position + 1],
            "past_key_values": prefix,
            "cache_position": torch.as_tensor(
                [token_position], device=device, dtype=torch.long,
            ),
            "use_cache": True,
            "return_dict": True,
        }
        with torch.inference_mode():
            output = model(**kwargs)
        next_prefix = output.past_key_values
        del output
        if prefix is not None and prefix is not next_prefix:
            del prefix
        prefix = next_prefix
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if prefix is None:
        raise RuntimeError("decode position must have a non-empty exact prefix")
    return prefix, elapsed


def _baseline_bundle(
    observer: MultiLayerCachedDecodeObserver,
    model: Any,
    encoded: Mapping[str, torch.Tensor],
    position: int,
    exact_prefix: Any,
    layers: Sequence[int],
) -> tuple[dict[int, DecodeObservation], dict[str, Any], float]:
    """Run one paired native token baseline and expose read-only layer views."""

    admitted = tuple(sorted(set(map(int, layers))))
    observer.set_injection_layer(admitted[0])
    first = observer.run(model, encoded, position, clone_decode_cache(exact_prefix))
    first_x = dict(observer.x_by_layer)
    first_routed = dict(observer.routed_by_layer)
    second = observer.run(model, encoded, position, clone_decode_cache(exact_prefix))
    second_x = dict(observer.x_by_layer)
    second_routed = dict(observer.routed_by_layer)
    maxima = _observation_maxima(first, second)
    cache = classified_cache_metrics(first.cache, second.cache)
    x_max_abs = max(
        float(torch.max(torch.abs(first_x[layer].float() - second_x[layer].float())).item())
        for layer in admitted
    )
    routed_max_abs = max(
        float(torch.max(torch.abs(
            first_routed[layer].float() - second_routed[layer].float()
        )).item())
        for layer in admitted
    )
    exact = bool(
        all(float(maxima[name]) == 0.0 for name in (
            "hidden_max_abs", "router_logits_max_abs", "router_scores_max_abs",
            "x_max_abs", "routed_max_abs", "terminal_logits_max_abs",
        ))
        and bool(maxima["router_ids_equal"])
        and bool(cache["full_cache_bit_identical"])
        and x_max_abs == 0.0
        and routed_max_abs == 0.0
    )
    if not exact:
        raise RuntimeError("paired request-major cached-decode baseline is not bit-identical")
    views = {
        layer: DecodeObservation(
            hidden=first.hidden,
            router_logits=first.router_logits,
            router_scores=first.router_scores,
            router_ids=first.router_ids,
            x=first_x[layer],
            routed=first_routed[layer],
            logits=first.logits,
            cache=first.cache,
            elapsed_seconds=first.elapsed_seconds,
        )
        for layer in admitted
    }
    facts = {
        **maxima,
        **{f"paired_baseline_{name}": value for name, value in cache.items()},
        "paired_baseline_all_injection_x_max_abs": x_max_abs,
        "paired_baseline_all_injection_routed_max_abs": routed_max_abs,
        "paired_baseline_all_exact": True,
        "paired_baseline_shared_across_injection_layers": True,
    }
    elapsed = first.elapsed_seconds + second.elapsed_seconds
    del second
    return views, facts, elapsed


def _baseline_pair(
    observer: CachedDecodeObserver,
    model: Any,
    encoded: Mapping[str, torch.Tensor],
    position: int,
    exact_prefix: Any,
) -> tuple[DecodeObservation, dict[str, Any], float]:
    first = observer.run(model, encoded, position, clone_decode_cache(exact_prefix))
    second = observer.run(model, encoded, position, clone_decode_cache(exact_prefix))
    maxima = _observation_maxima(first, second)
    cache = classified_cache_metrics(first.cache, second.cache)
    exact = bool(
        all(float(maxima[name]) == 0.0 for name in (
            "hidden_max_abs", "router_logits_max_abs", "router_scores_max_abs",
            "x_max_abs", "routed_max_abs", "terminal_logits_max_abs",
        ))
        and bool(maxima["router_ids_equal"])
        and bool(cache["full_cache_bit_identical"])
    )
    if not exact:
        raise RuntimeError("paired native cached-decode baseline is not bit-identical")
    elapsed = first.elapsed_seconds + second.elapsed_seconds
    del second
    return first, {
        **maxima,
        **{f"paired_baseline_{name}": value for name, value in cache.items()},
        "paired_baseline_all_exact": True,
    }, elapsed


def _prepare_layer_experts(
    checkpoint: Path,
    trees_path: Path,
    layer: int,
    expert_ids: Sequence[int],
    workers: int,
) -> dict[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]]:
    index = load_json(Path(checkpoint) / "model.safetensors.index.json")["weight_map"]
    tree_records = load_json(trees_path)
    trees = {name: tree_from_record(tree_records[name]) for name in pr13.PROJECTIONS}
    active = sorted(set(map(int, expert_ids)))
    if not active:
        raise RuntimeError("sealed layer contains no routed experts")

    def prepare(expert: int):
        decoded = set_study.decode_expert(
            checkpoint, index, trees, int(layer), int(expert),
        )
        q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
        q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
        return int(expert), (q2, q4)

    result: dict[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] = {}
    with ThreadPoolExecutor(max_workers=min(max(1, int(workers)), len(active))) as executor:
        futures = {executor.submit(prepare, expert): expert for expert in active}
        for completed, future in enumerate(as_completed(futures), start=1):
            expert, decoded = future.result()
            result[expert] = decoded
            if completed % 16 == 0 or completed == len(active):
                print(
                    f"[nested outcomes layer {layer}] decoded {completed}/{len(active)} experts",
                    flush=True,
                )
    return result


def _selected_state_delta(
    allocation: OutcomeAllocation,
    baseline: DecodeObservation,
    decoded: Mapping[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
    responses: Sequence[Any],
    *,
    q4_routed_max_abs_atol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    layer = int(allocation.layer)
    source_ids = np.asarray(allocation.expert_ids, np.int64)
    live_ids = baseline.router_ids[layer].reshape(-1).numpy().astype(np.int64)
    if source_ids.shape != (8,) or live_ids.shape != (8,):
        raise RuntimeError("routed expert identity must contain eight values")
    if set(source_ids.tolist()) != set(live_ids.tolist()):
        raise RuntimeError(
            f"sealed allocation route differs from exact full-model route: "
            f"allocation={allocation.identity} source={source_ids.tolist()} "
            f"live={live_ids.tolist()}"
        )
    source_rank = {int(expert): rank for rank, expert in enumerate(source_ids)}
    states = np.stack([
        allocation.selected_state[source_rank[int(expert)]]
        for expert in live_ids
    ])
    if len(responses) != 8:
        raise RuntimeError("live expert response set must contain eight experts")
    weights = _np(baseline.router_scores[layer]).reshape(-1).astype(np.float64)
    if weights.shape != (8,) or np.any(weights <= 0.0):
        raise RuntimeError("live execution router weights are invalid")
    targets = []
    approximate = []
    for response, state in zip(responses, states, strict=True):
        targets.append(np.asarray(response.target_output, np.float64))
        approximate.append(np.asarray(split_state_output(response, state), np.float64))
    targets_array = np.stack(targets)
    approximate_array = np.stack(approximate)
    delta = np.einsum(
        "e,eo->o", weights, approximate_array - targets_array, optimize=True,
    )
    q4_routed = np.einsum("e,eo->o", weights, targets_array, optimize=True)
    reference = _np(baseline.routed).reshape(-1).astype(np.float64)
    q4_max_abs = float(np.max(np.abs(q4_routed - reference), initial=0.0))
    if q4_max_abs > float(q4_routed_max_abs_atol):
        raise RuntimeError(
            f"live Q4 reconstruction exceeds tolerance: {q4_max_abs} > "
            f"{q4_routed_max_abs_atol}"
        )
    live = np.asarray(delta, np.float32)
    return live, {
        "selected_group_pages": int(allocation.selected_pages),
        "selected_page_cap": int(allocation.page_cap),
        "live_injected_delta_mse": float(np.mean(live.astype(np.float64) ** 2)),
        "live_q4_routed_max_abs": q4_max_abs,
        "source_and_live_route_set_equal": True,
        "source_and_live_route_order_equal": bool(np.array_equal(source_ids, live_ids)),
        "live_execution_router_weight_sum": float(weights.sum()),
        "delta_execution": "sealed_complete_state_reconstructed_on_live_cached_activation",
        "activation_dependent_page_effects": True,
    }


def _base_row(
    allocation: OutcomeAllocation,
    request: OutcomeRequest,
    route_mode: str,
    allocation_manifest_sha256: str,
    zero_dose_gate_id: str,
) -> dict[str, Any]:
    return {
        "schema": OUTCOME_SCHEMA,
        "split": allocation.split,
        "domain": request.domain,
        "prompt_sha256": request.prompt_sha256,
        "arm": allocation.arm,
        "rate_pages_per_expert": allocation.rate,
        "injection_layer": allocation.layer,
        "request_id": allocation.request_id,
        "position": request.position,
        "prefix_tokens": request.position,
        "next_token_id": request.next_token_id,
        "route_mode": route_mode,
        "selected_state_sha256": allocation.state_sha256,
        "allocation_manifest_sha256": allocation_manifest_sha256,
        "zero_dose_gate_id": zero_dose_gate_id,
        "query_length": 1,
        "cross_position_delta_coupling": False,
        "candidate_cache_reused": False,
    }


def _zero_dose_gate_id(
    *,
    allocation_manifest_sha256: str,
    split: str,
    layer: int,
    request_id: str,
    route_mode: str,
) -> str:
    return canonical_sha256({
        "schema": "pr13_d1_nested_zero_dose_gate_identity_v1",
        "allocation_manifest_sha256": allocation_manifest_sha256,
        "split": split,
        "injection_layer": int(layer),
        "request_id": str(request_id),
        "route_mode": str(route_mode),
    })


def _prove_zero_frozen_routes_are_identity(
    baseline: DecodeObservation,
    layer: int,
    device: torch.device,
) -> dict[str, Any]:
    """Prove both frozen controls are tensor identities at zero dose.

    Fully frozen replay returns the exact captured IDs and scores by
    construction.  For frozen-set/live-weight replay, reproduce the hook's
    softmax/gather/renormalize computation and require bitwise equality to the
    native captured execution scores at every downstream layer.
    """

    maximum = 0.0
    for downstream in range(int(layer) + 1, 40):
        logits = baseline.router_logits[downstream].to(device)
        ids = baseline.router_ids[downstream].to(device)
        native = baseline.router_scores[downstream]
        probabilities = torch.softmax(logits, dtype=torch.float, dim=-1)
        recomputed = torch.gather(
            probabilities, dim=-1, index=ids,
        )
        recomputed = recomputed / recomputed.sum(dim=-1, keepdim=True)
        recomputed = recomputed.to(native.dtype)
        difference = float(torch.max(torch.abs(
            recomputed.float() - native.float()
        )).item())
        maximum = max(maximum, difference)
        if not torch.equal(recomputed.cpu(), native.cpu()):
            raise RuntimeError(
                "zero-dose frozen-set/live-weight hook is not a bitwise identity "
                f"at downstream layer {downstream}"
            )
    return {
        "zero_fully_frozen_route_hook_identity_by_construction": True,
        "zero_frozen_set_live_weight_all_downstream_scores_bit_identical": True,
        "zero_frozen_set_live_weight_max_abs": maximum,
    }


def run_cell(
    *,
    model: Any,
    split: str,
    layer: int,
    allocations: Sequence[OutcomeAllocation],
    requests: Mapping[str, OutcomeRequest],
    decoded: Mapping[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
    output: Path,
    pins: Mapping[str, Any],
    q4_routed_max_abs_atol: float,
) -> None:
    paths = _cell_paths(output, split, layer)
    if _completed_cell(
        paths, split=split, layer=layer, pins=pins, allocations=allocations,
    ):
        print(f"[resume nested outcomes] split={split} layer={layer}", flush=True)
        return
    by_request: dict[str, list[OutcomeAllocation]] = {}
    for allocation in allocations:
        if allocation.split != split or allocation.layer != layer:
            raise ValueError("run_cell received an allocation outside its cell")
        by_request.setdefault(allocation.request_id, []).append(allocation)
    if set(by_request) != {
        request_id for request_id, request in requests.items() if request.split == split
    }:
        raise RuntimeError("outcome cell request set differs from the sealed grid")

    observer = CachedDecodeObserver(model, layer)
    quality_rows: list[dict[str, Any]] = []
    propagation_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    prefix_seconds = 0.0
    baseline_seconds = 0.0
    candidate_seconds = 0.0
    physical_allocation_states = 0
    physical_candidate_executions = 0
    logical_candidate_comparisons = 0
    zero_candidate_executions = 0
    device = model.get_input_embeddings().weight.device
    try:
        for request_number, request_id in enumerate(sorted(by_request), start=1):
            request = requests[request_id]
            encoded = _encoded_request(request, device)
            exact_prefix, elapsed = advance_exact_prefix(
                observer, model, encoded, request.position,
            )
            prefix_seconds += elapsed
            baseline, repeat, elapsed = _baseline_pair(
                observer, model, encoded, request.position, exact_prefix,
            )
            baseline_seconds += elapsed

            for route_mode in ROUTE_MODES:
                zero_delta = np.zeros(int(baseline.routed.numel()), np.float32)
                zero_gate_id = _zero_dose_gate_id(
                    allocation_manifest_sha256=str(pins["allocation_manifest_sha256"]),
                    split=split,
                    layer=layer,
                    request_id=request_id,
                    route_mode=route_mode,
                )
                zero_candidate, injection = _run_candidate(
                    observer, model, encoded, request.position, exact_prefix,
                    baseline, layer, zero_delta, route_mode,
                )
                zero_candidate_executions += 1
                candidate_seconds += zero_candidate.elapsed_seconds
                zero_metric = _zero_metrics(baseline, zero_candidate)
                zero_rows.append({
                    "schema": OUTCOME_SCHEMA,
                    "split": split,
                    "domain": request.domain,
                    "request_id": request_id,
                    "position": request.position,
                    "injection_layer": layer,
                    "route_mode": route_mode,
                    "zero_dose_gate_id": zero_gate_id,
                    "logical_allocations_authenticated": len(by_request[request_id]),
                    "routed_repeat_max_abs": injection["routed_repeat_max_abs"],
                    "paired_baseline_full_cache_bit_identical": repeat[
                        "paired_baseline_full_cache_bit_identical"
                    ],
                    **zero_metric,
                })
                del zero_candidate

            baseline_hidden = baseline.numpy_hidden()
            baseline_router = baseline.numpy_router()
            live_ids = baseline.router_ids[layer].reshape(-1).numpy().astype(np.int64)
            responses = projection_responses(
                _np(baseline.x).reshape(-1), live_ids, decoded,
            )
            prepared = []
            for allocation in sorted(
                by_request[request_id], key=lambda row: (row.arm, row.rate),
            ):
                delta, reconstruction = _selected_state_delta(
                    allocation, baseline, decoded, responses,
                    q4_routed_max_abs_atol=q4_routed_max_abs_atol,
                )
                prepared.append({
                    "allocation": allocation,
                    "delta": delta,
                    "reconstruction": reconstruction,
                    **physical_execution_identity(allocation, delta),
                })
            physical_groups: dict[str, list[dict[str, Any]]] = {}
            for item in prepared:
                physical_groups.setdefault(
                    str(item["physical_execution_id"]), [],
                ).append(item)
            physical_allocation_states += len(physical_groups)
            for execution_id in sorted(physical_groups):
                members = sorted(
                    physical_groups[execution_id],
                    key=lambda item: (
                        item["allocation"].arm, item["allocation"].rate,
                    ),
                )
                representative = members[0]
                delta = np.asarray(representative["delta"], np.float32)
                if any(
                    item["physical_expert_state_sha256"]
                    != representative["physical_expert_state_sha256"]
                    or item["physical_delta_sha256"]
                    != representative["physical_delta_sha256"]
                    or not np.array_equal(np.asarray(item["delta"], np.float32), delta)
                    for item in members[1:]
                ):
                    raise RuntimeError("physical execution identity collision")
                fanout = len(members)
                for route_mode in ROUTE_MODES:
                    zero_gate_id = _zero_dose_gate_id(
                        allocation_manifest_sha256=str(
                            pins["allocation_manifest_sha256"]
                        ),
                        split=split,
                        layer=layer,
                        request_id=request_id,
                        route_mode=route_mode,
                    )
                    candidate, injection = _run_candidate(
                        observer, model, encoded, request.position, exact_prefix,
                        baseline, layer, delta, route_mode,
                    )
                    physical_candidate_executions += 1
                    candidate_seconds += candidate.elapsed_seconds
                    candidate_hidden = candidate.numpy_hidden()
                    candidate_router = candidate.numpy_router()
                    quality = current_token_quality_metrics(
                        _np(baseline.logits), _np(candidate.logits),
                        request.next_token_id, baseline_hidden[39], candidate_hidden[39],
                    )
                    routes, first_changed = downstream_route_rows(
                        baseline_router, candidate_router,
                        baseline_hidden, candidate_hidden, layer,
                    )
                    cache = classified_cache_metrics(baseline.cache, candidate.cache)
                    route_and_cache = []
                    for route in routes:
                        observation_layer = int(route["observation_layer"])
                        route_and_cache.append((
                            route,
                            cache_state_metrics(
                                baseline.cache, candidate.cache, observation_layer,
                            ).to_dict("post_token_layer_cache"),
                        ))
                    for ordinal, item in enumerate(members):
                        allocation = item["allocation"]
                        reconstruction = item["reconstruction"]
                        logical_candidate_comparisons += 1
                        provenance = {
                            "physical_execution_id": execution_id,
                            "physical_expert_state_sha256": item[
                                "physical_expert_state_sha256"
                            ],
                            "physical_delta_sha256": item["physical_delta_sha256"],
                            "physical_execution_logical_fanout": fanout,
                            "physical_execution_reuse_ordinal": ordinal,
                            "physical_execution_reused": ordinal > 0,
                            "physical_representative_arm": representative[
                                "allocation"
                            ].arm,
                            "physical_representative_rate_pages_per_expert": representative[
                                "allocation"
                            ].rate,
                            "physical_representative_selected_state_sha256": representative[
                                "allocation"
                            ].state_sha256,
                        }
                        base = {
                            **_base_row(
                                allocation, request, route_mode,
                                str(pins["allocation_manifest_sha256"]),
                                zero_gate_id,
                            ),
                            **provenance,
                        }
                        quality_rows.append({
                            **base,
                            **reconstruction,
                            **quality,
                            "realized_injected_layer_output_mse": float(np.mean(
                                (baseline_hidden[layer].astype(np.float64)
                                 - candidate_hidden[layer].astype(np.float64)) ** 2
                            )),
                            "first_route_membership_change_layer": (
                                int(first_changed) if first_changed is not None else -1
                            ),
                            "routed_repeat_max_abs": injection[
                                "routed_repeat_max_abs"
                            ],
                            "physical_candidate_seconds": candidate.elapsed_seconds,
                            "logical_amortized_candidate_seconds": (
                                candidate.elapsed_seconds / fanout
                            ),
                        })
                        cache_rows.append({**base, **cache})
                        for route, layer_cache in route_and_cache:
                            propagation_rows.append({
                                **base,
                                **route,
                                **layer_cache,
                                "router_observation_semantics": (
                                    "live_provisional_logits_before_any_downstream_route_freeze"
                                ),
                            })
                    del candidate, candidate_hidden, candidate_router
            print(
                f"[nested outcomes] split={split} layer={layer} "
                f"request={request_number}/{len(by_request)}",
                flush=True,
            )
            del responses, baseline_router, baseline_hidden, baseline, exact_prefix, encoded
            gc.collect()
    finally:
        observer.close()

    expected_quality = len(allocations) * len(ROUTE_MODES)
    expected_propagation = expected_quality * (39 - int(layer))
    expected_zero = len(by_request) * len(ROUTE_MODES)
    if len(quality_rows) != expected_quality:
        raise RuntimeError("nested outcome quality grid is incomplete")
    if len(propagation_rows) != expected_propagation:
        raise RuntimeError("nested outcome propagation grid is incomplete")
    if len(cache_rows) != expected_quality:
        raise RuntimeError("nested outcome cache grid is incomplete")
    if len(zero_rows) != expected_zero:
        raise RuntimeError("nested outcome zero-dose grid is incomplete")
    require_zero_dose_parity(pd.DataFrame(zero_rows))

    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_parquet(paths["quality"], quality_rows)
    atomic_parquet(paths["propagation"], propagation_rows)
    atomic_parquet(paths["cache"], cache_rows)
    atomic_parquet(paths["zero"], zero_rows)
    files = {
        paths[key].name: {
            "sha256": file_sha256(paths[key]),
            "bytes": paths[key].stat().st_size,
            "rows": len(rows),
        }
        for key, rows in (
            ("quality", quality_rows),
            ("propagation", propagation_rows),
            ("cache", cache_rows),
            ("zero", zero_rows),
        )
    }
    atomic_json(paths["facts"], {
        "schema": OUTCOME_SCHEMA,
        "completed": True,
        "split": split,
        "injection_layer": layer,
        "requests": len(by_request),
        "allocations": len(allocations),
        "route_modes": list(ROUTE_MODES),
        "quality_rows": len(quality_rows),
        "propagation_rows": len(propagation_rows),
        "cache_rows": len(cache_rows),
        "zero_rows": len(zero_rows),
        "prefix_seconds": prefix_seconds,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "input_pins": dict(pins),
        "files": files,
        "exact_prefix_cache_cloned_per_candidate": True,
        "candidate_cache_reused": False,
        "current_token_only": True,
        "zero_dose_all_cache_commits_exact": True,
        "allocation_selected_from_terminal_outcomes": False,
        "experiment_b_started": False,
    })


def _new_cell_buffer(
    split: str,
    layer: int,
    allocations: Sequence[OutcomeAllocation],
) -> dict[str, Any]:
    by_request: dict[str, tuple[OutcomeAllocation, ...]] = {}
    request_ids = sorted({row.request_id for row in allocations})
    for request_id in request_ids:
        rows = tuple(
            sorted(
                (
                    row for row in allocations
                    if row.request_id == request_id
                ),
                key=lambda row: (row.arm, row.rate),
            )
        )
        if not rows:
            raise RuntimeError("cell buffer has an empty request allocation set")
        if any(row.split != split or row.layer != int(layer) for row in rows):
            raise RuntimeError("cell buffer received an allocation outside its cell")
        by_request[request_id] = rows
    return {
        "split": str(split),
        "layer": int(layer),
        "allocations": tuple(allocations),
        "by_request": by_request,
        "quality_rows": [],
        "propagation_rows": [],
        "cache_rows": [],
        "zero_rows": [],
        "prefix_seconds_amortized": 0.0,
        "baseline_seconds_amortized": 0.0,
        "candidate_seconds": 0.0,
        "physical_allocation_states": 0,
        "physical_candidate_executions": 0,
        "logical_candidate_comparisons": 0,
        "zero_candidate_executions": 0,
        "requests_executed": 0,
    }


def _execute_request_layer(
    *,
    observer: MultiLayerCachedDecodeObserver,
    model: Any,
    encoded: Mapping[str, torch.Tensor],
    exact_prefix: Any,
    baseline: DecodeObservation,
    repeat: Mapping[str, Any],
    request: OutcomeRequest,
    buffer: dict[str, Any],
    decoded: Mapping[int, tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
    pins: Mapping[str, Any],
    q4_routed_max_abs_atol: float,
) -> None:
    """Execute one request/layer from a request-shared exact prefix/baseline."""

    split = str(buffer["split"])
    layer = int(buffer["layer"])
    allocations = buffer["by_request"].get(request.request_id)
    if allocations is None:
        raise RuntimeError("request is absent from the sealed cell allocation grid")
    observer.set_injection_layer(layer)
    zero_delta = np.zeros(int(baseline.routed.numel()), np.float32)
    frozen_identity = _prove_zero_frozen_routes_are_identity(
        baseline, layer, encoded["input_ids"].device,
    )
    for route_mode in ROUTE_MODES:
        zero_gate_id = _zero_dose_gate_id(
            allocation_manifest_sha256=str(pins["allocation_manifest_sha256"]),
            split=split,
            layer=layer,
            request_id=request.request_id,
            route_mode=route_mode,
        )
        zero_candidate, injection = _run_candidate(
            observer, model, encoded, request.position, exact_prefix,
            baseline, layer, zero_delta, route_mode,
        )
        buffer["zero_candidate_executions"] += 1
        buffer["candidate_seconds"] += zero_candidate.elapsed_seconds
        zero_metric = _zero_metrics(baseline, zero_candidate)
        buffer["zero_rows"].append({
            "schema": OUTCOME_SCHEMA,
            "split": split,
            "domain": request.domain,
            "request_id": request.request_id,
            "position": request.position,
            "injection_layer": layer,
            "route_mode": route_mode,
            "zero_dose_gate_id": zero_gate_id,
            "zero_physical_execution_mode": route_mode,
            "zero_physical_execution_was_run": True,
            "logical_allocations_authenticated": len(allocations),
            "routed_repeat_max_abs": injection["routed_repeat_max_abs"],
            "paired_baseline_full_cache_bit_identical": repeat[
                "paired_baseline_full_cache_bit_identical"
            ],
            "paired_baseline_shared_across_injection_layers": repeat[
                "paired_baseline_shared_across_injection_layers"
            ],
            **frozen_identity,
            **zero_metric,
        })
        del zero_candidate

    baseline_hidden = baseline.numpy_hidden()
    baseline_router = baseline.numpy_router()
    live_ids = baseline.router_ids[layer].reshape(-1).numpy().astype(np.int64)
    responses = projection_responses(
        _np(baseline.x).reshape(-1), live_ids, decoded,
    )
    prepared: list[dict[str, Any]] = []
    for allocation in allocations:
        delta, reconstruction = _selected_state_delta(
            allocation, baseline, decoded, responses,
            q4_routed_max_abs_atol=q4_routed_max_abs_atol,
        )
        prepared.append({
            "allocation": allocation,
            "delta": delta,
            "reconstruction": reconstruction,
            **physical_execution_identity(allocation, delta),
        })
    physical_groups: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        physical_groups.setdefault(str(item["physical_execution_id"]), []).append(item)
    buffer["physical_allocation_states"] += len(physical_groups)
    for execution_id in sorted(physical_groups):
        members = sorted(
            physical_groups[execution_id],
            key=lambda item: (item["allocation"].arm, item["allocation"].rate),
        )
        representative = members[0]
        delta = np.asarray(representative["delta"], np.float32)
        if any(
            item["physical_expert_state_sha256"]
            != representative["physical_expert_state_sha256"]
            or item["physical_delta_sha256"]
            != representative["physical_delta_sha256"]
            or not np.array_equal(np.asarray(item["delta"], np.float32), delta)
            for item in members[1:]
        ):
            raise RuntimeError("physical execution identity collision")
        fanout = len(members)
        for route_mode in ROUTE_MODES:
            zero_gate_id = _zero_dose_gate_id(
                allocation_manifest_sha256=str(pins["allocation_manifest_sha256"]),
                split=split,
                layer=layer,
                request_id=request.request_id,
                route_mode=route_mode,
            )
            candidate, injection = _run_candidate(
                observer, model, encoded, request.position, exact_prefix,
                baseline, layer, delta, route_mode,
            )
            buffer["physical_candidate_executions"] += 1
            buffer["candidate_seconds"] += candidate.elapsed_seconds
            candidate_hidden = candidate.numpy_hidden()
            candidate_router = candidate.numpy_router()
            quality = current_token_quality_metrics(
                _np(baseline.logits), _np(candidate.logits), request.next_token_id,
                baseline_hidden[39], candidate_hidden[39],
            )
            routes, first_changed = downstream_route_rows(
                baseline_router, candidate_router,
                baseline_hidden, candidate_hidden, layer,
            )
            cache = classified_cache_metrics(baseline.cache, candidate.cache)
            route_and_cache = [
                (
                    route,
                    cache_state_metrics(
                        baseline.cache, candidate.cache,
                        int(route["observation_layer"]),
                    ).to_dict("post_token_layer_cache"),
                )
                for route in routes
            ]
            for ordinal, item in enumerate(members):
                allocation = item["allocation"]
                reconstruction = item["reconstruction"]
                buffer["logical_candidate_comparisons"] += 1
                provenance = {
                    "physical_execution_id": execution_id,
                    "physical_expert_state_sha256": item[
                        "physical_expert_state_sha256"
                    ],
                    "physical_delta_sha256": item["physical_delta_sha256"],
                    "physical_execution_logical_fanout": fanout,
                    "physical_execution_reuse_ordinal": ordinal,
                    "physical_execution_reused": ordinal > 0,
                    "physical_representative_arm": representative["allocation"].arm,
                    "physical_representative_rate_pages_per_expert": (
                        representative["allocation"].rate
                    ),
                    "physical_representative_selected_state_sha256": (
                        representative["allocation"].state_sha256
                    ),
                }
                base = {
                    **_base_row(
                        allocation, request, route_mode,
                        str(pins["allocation_manifest_sha256"]), zero_gate_id,
                    ),
                    **provenance,
                }
                buffer["quality_rows"].append({
                    **base,
                    **reconstruction,
                    **quality,
                    "realized_injected_layer_output_mse": float(np.mean(
                        (
                            baseline_hidden[layer].astype(np.float64)
                            - candidate_hidden[layer].astype(np.float64)
                        ) ** 2
                    )),
                    "first_route_membership_change_layer": (
                        int(first_changed) if first_changed is not None else -1
                    ),
                    "routed_repeat_max_abs": injection["routed_repeat_max_abs"],
                    "physical_candidate_seconds": candidate.elapsed_seconds,
                    "logical_amortized_candidate_seconds": (
                        candidate.elapsed_seconds / fanout
                    ),
                })
                buffer["cache_rows"].append({**base, **cache})
                for route, layer_cache in route_and_cache:
                    buffer["propagation_rows"].append({
                        **base,
                        **route,
                        **layer_cache,
                        "router_observation_semantics": (
                            "live_provisional_logits_before_any_downstream_route_freeze"
                        ),
                    })
            del candidate, candidate_hidden, candidate_router
    buffer["requests_executed"] += 1
    del responses, baseline_router, baseline_hidden


def _write_cell_buffer(
    *,
    buffer: Mapping[str, Any],
    output: Path,
    pins: Mapping[str, Any],
) -> None:
    split = str(buffer["split"])
    layer = int(buffer["layer"])
    allocations = tuple(buffer["allocations"])
    by_request = buffer["by_request"]
    rows = {
        "quality": list(buffer["quality_rows"]),
        "propagation": list(buffer["propagation_rows"]),
        "cache": list(buffer["cache_rows"]),
        "zero": list(buffer["zero_rows"]),
    }
    expected = {
        "quality": len(allocations) * len(ROUTE_MODES),
        "propagation": len(allocations) * len(ROUTE_MODES) * (39 - layer),
        "cache": len(allocations) * len(ROUTE_MODES),
        "zero": len(by_request) * len(ROUTE_MODES),
    }
    observed = {key: len(value) for key, value in rows.items()}
    if observed != expected:
        raise RuntimeError(f"request-major outcome cell is incomplete: {observed} != {expected}")
    if int(buffer["requests_executed"]) != len(by_request):
        raise RuntimeError("request-major outcome cell request coverage is incomplete")
    if int(buffer["logical_candidate_comparisons"]) != expected["quality"]:
        raise RuntimeError("logical candidate comparison count changed")
    if int(buffer["zero_candidate_executions"]) != expected["zero"]:
        raise RuntimeError("zero-dose physical execution count changed")
    if int(buffer["physical_candidate_executions"]) != (
        int(buffer["physical_allocation_states"]) * len(ROUTE_MODES)
    ):
        raise RuntimeError("physical candidate execution count changed")
    if int(buffer["physical_candidate_executions"]) > expected["quality"]:
        raise RuntimeError("physical deduplication executed more than the logical grid")
    frames = {key: pd.DataFrame(value) for key, value in rows.items()}
    _validate_cell_tables(
        frames, split=split, layer=layer, allocations=allocations,
    )
    paths = _cell_paths(output, split, layer)
    paths["root"].mkdir(parents=True, exist_ok=True)
    for key in ("quality", "propagation", "cache", "zero"):
        atomic_parquet(paths[key], rows[key])
    files = {
        paths[key].name: {
            "sha256": file_sha256(paths[key]),
            "bytes": paths[key].stat().st_size,
            "rows": len(rows[key]),
        }
        for key in ("quality", "propagation", "cache", "zero")
    }
    logical = int(buffer["logical_candidate_comparisons"])
    physical = int(buffer["physical_candidate_executions"])
    atomic_json(paths["facts"], {
        "schema": OUTCOME_SCHEMA,
        "completed": True,
        "split": split,
        "injection_layer": layer,
        "requests": len(by_request),
        "allocations": len(allocations),
        "route_modes": list(ROUTE_MODES),
        "quality_rows": observed["quality"],
        "propagation_rows": observed["propagation"],
        "cache_rows": observed["cache"],
        "zero_rows": observed["zero"],
        "request_major_prefix_seconds_amortized": float(
            buffer["prefix_seconds_amortized"]
        ),
        "request_major_baseline_seconds_amortized": float(
            buffer["baseline_seconds_amortized"]
        ),
        "candidate_seconds": float(buffer["candidate_seconds"]),
        "physical_allocation_states": int(buffer["physical_allocation_states"]),
        "physical_candidate_executions": physical,
        "logical_candidate_comparisons": logical,
        "zero_candidate_executions": int(buffer["zero_candidate_executions"]),
        "logical_zero_dose_gates": observed["zero"],
        "every_zero_route_mode_physically_executed": True,
        "physical_execution_fraction_of_logical": physical / logical,
        "input_pins": dict(pins),
        "files": files,
        "one_exact_pretoken_cache_per_request_across_all_layers": True,
        "one_paired_native_baseline_per_request_across_all_layers": True,
        "exact_prefix_cache_cloned_per_physical_candidate": True,
        "logical_rows_fanned_out_from_physical_execution": True,
        "candidate_cache_reused": False,
        "current_token_only": True,
        "zero_dose_all_cache_commits_exact": True,
        "allocation_selected_from_terminal_outcomes": False,
        "experiment_b_started": False,
    })


def _selected_allocations(
    plan: Any,
    splits: Sequence[str],
    layers: Sequence[int] | None,
) -> tuple[OutcomeAllocation, ...]:
    selected_splits = set(map(str, splits))
    if not selected_splits or not selected_splits.issubset({"calibration", "evaluation"}):
        raise ValueError("splits must be calibration and/or evaluation")
    selected_layers = None if layers is None else set(map(int, layers))
    selected = tuple(
        row for row in plan.allocations
        if row.split in selected_splits
        and (selected_layers is None or row.layer in selected_layers)
    )
    if not selected:
        raise ValueError("no sealed allocations match the requested outcome grid")
    validate_complete_allocation_grid(selected)
    return selected


def _resolve_request_manifest_facts_path(
    capture_config: Mapping[str, Any],
    manifest_path: Path,
    supplied_path: Path | None,
) -> Path:
    """Resolve facts without parsing or exposing any request token IDs."""

    if supplied_path is not None:
        return Path(supplied_path)
    manifest_path = Path(manifest_path)
    sibling = manifest_path.with_name(f"{manifest_path.stem}_facts.json")
    if sibling.is_file():
        return sibling
    request_source = capture_config.get("request_source")
    if isinstance(request_source, Mapping):
        configured = request_source.get("selected_manifest_facts")
        if isinstance(configured, str) and configured:
            configured_path = Path(configured)
            if configured_path.is_file():
                return configured_path
    raise FileNotFoundError(
        "selected request-manifest facts are required; supply "
        "--request-manifest-facts"
    )


def _authenticate_raw_request_inputs(
    plan: Any,
    *,
    manifest_path: Path,
    manifest_facts_path: Path,
) -> dict[str, str]:
    """Bind raw request bytes to calibration before any token row is parsed."""

    observed_manifest = file_sha256(Path(manifest_path))
    if observed_manifest != str(plan.request_manifest_sha256):
        raise ValueError(
            "supplied request manifest does not match the frozen calibration "
            "raw SHA-256"
        )
    observed_facts = file_sha256(Path(manifest_facts_path))
    if observed_facts != str(plan.request_manifest_facts_sha256):
        raise ValueError(
            "supplied request manifest facts do not match the frozen calibration "
            "raw SHA-256"
        )
    return {
        "manifest_sha256": observed_manifest,
        "manifest_facts_sha256": observed_facts,
    }


def _load_inputs(args: argparse.Namespace):
    allocation_config_sha256 = file_sha256(args.config)
    frozen_config_sha256 = file_sha256(FROZEN_ALLOCATION_CONFIG_PATH)
    if frozen_config_sha256 != ALLOCATION_CONFIG_FILE_SHA256:
        raise RuntimeError("checked-in immutable v2 allocation config bytes changed")
    if allocation_config_sha256 != ALLOCATION_CONFIG_FILE_SHA256:
        raise ValueError(
            "supplied allocation config bytes differ from the frozen checked-in config"
        )
    config = load_json(args.config)
    validate_outcome_config(config)
    capture_pin = config["authenticated_capture_protocol"]
    capture_path = EXPERIMENT / str(capture_pin["path"])
    if (
        str(capture_pin["path"]) != AUTHENTICATED_CAPTURE_CONFIG_PATH
        or str(capture_pin["sha256"]) != AUTHENTICATED_CAPTURE_CONFIG_SHA256
        or not capture_path.is_file()
        or file_sha256(capture_path) != AUTHENTICATED_CAPTURE_CONFIG_SHA256
    ):
        raise ValueError("authenticated capture protocol bytes changed")
    capture_config = load_json(capture_path)
    capture_critical_fields = (
        "checkpoint", "checkpoint_revision", "checkpoint_index_sha256",
        "checkpoint_config_sha256", "tokenizer_json_sha256", "selected_trees",
        "selected_tree_sha256", "request_source", "decode_position",
        "injection_layers",
    )
    changed = [
        field for field in capture_critical_fields
        if config.get(field) != capture_config.get(field)
    ]
    if changed:
        raise ValueError(
            f"allocation config differs from authenticated capture protocol: {changed}"
        )
    plan = load_outcome_plan(
        manifest_path=args.allocation_manifest,
        seal_path=args.allocation_seal,
        expected_manifest_sha256=args.expected_allocation_sha256,
        expected_frozen_calibration_spec_sha256=args.expected_calibration_sha256,
    )
    validate_plan_against_config(
        plan, config, config_file_sha256=allocation_config_sha256,
    )
    compatibility_path = Path(args.outcome_compatibility_protocol)
    compatibility_sha256 = file_sha256(compatibility_path)
    if compatibility_sha256 != args.expected_compatibility_sha256:
        raise ValueError("outcome compatibility protocol SHA-256 changed")
    compatibility_protocol = load_json(compatibility_path)
    if compatibility_protocol.get("schema") != OUTCOME_COMPATIBILITY_SCHEMA:
        raise ValueError("outcome compatibility protocol schema changed")
    outcome_code_bundle = _authenticate_outcome_code_bundle(
        plan,
        compatibility_protocol,
        allocation_config_sha256=allocation_config_sha256,
        capture_config=capture_config,
    )
    outcome_code_bundle["compatibility_protocol_file"] = {
        "path": str(compatibility_path),
        "sha256": compatibility_sha256,
    }
    request_facts_path = _resolve_request_manifest_facts_path(
        capture_config, args.request_manifest, args.request_manifest_facts,
    )
    authenticated_raw_hashes = _authenticate_raw_request_inputs(
        plan,
        manifest_path=args.request_manifest,
        manifest_facts_path=request_facts_path,
    )
    validated_capture_config, request_rows, request_hashes = validate_capture_inputs(
        config_path=capture_path,
        manifest_path=args.request_manifest,
        manifest_facts_path=request_facts_path,
    )
    if validated_capture_config != capture_config:
        raise RuntimeError("authenticated capture config changed during validation")
    if any(
        str(request_hashes[key]) != authenticated_raw_hashes[key]
        for key in ("manifest_sha256", "manifest_facts_sha256")
    ):
        raise RuntimeError("request inputs changed during authenticated validation")
    allocations = _selected_allocations(plan, args.splits, args.layers)
    requests = join_request_manifest(allocations, request_rows)
    return (
        config, plan, allocations, requests, request_hashes,
        allocation_config_sha256, outcome_code_bundle,
    )


def run_phase(args: argparse.Namespace) -> None:
    (
        config, plan, allocations, requests, request_hashes,
        allocation_config_sha256, outcome_code_bundle,
    ) = _load_inputs(args)
    hardware = _gpu_gate(config)
    execution_stack = _validate_execution_stack(config, hardware)
    checkpoint_hashes = validate_checkpoint_pins(config, args.checkpoint)
    tree_sha = file_sha256(args.trees)
    if tree_sha != str(config["selected_tree_sha256"]):
        raise RuntimeError("selected-tree artifact differs from the frozen config")
    pins = _input_pins(
        allocation_config_sha256=allocation_config_sha256,
        capture_hashes=request_hashes,
        checkpoint_hashes=checkpoint_hashes,
        selected_tree_sha256=tree_sha,
        allocation_facts=outcome_input_facts(plan),
        execution_stack=execution_stack,
        outcome_code_bundle=outcome_code_bundle,
        q4_routed_max_abs_atol=Q4_ROUTED_MAX_ABS_ATOL,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cell_buffers: dict[tuple[str, int], dict[str, Any]] = {}
    for split in args.splits:
        for layer in sorted({
            row.layer for row in allocations if row.split == split
        }):
            cell_allocations = tuple(
                row for row in allocations
                if row.split == split and row.layer == layer
            )
            paths = _cell_paths(args.output, split, layer)
            if _completed_cell(
                paths, split=split, layer=layer, pins=pins,
                allocations=cell_allocations,
            ):
                print(
                    f"[resume nested outcomes] split={split} layer={layer}",
                    flush=True,
                )
                continue
            cell_buffers[(split, layer)] = _new_cell_buffer(
                split, layer, cell_allocations,
            )
    if not cell_buffers:
        print("[resume nested outcomes] all selected cells are complete", flush=True)
        return

    model, tokenizer, load_seconds = _load_model(args.checkpoint)
    del tokenizer
    atomic_json(args.output / HOST_FACTS, {
        "schema": OUTCOME_SCHEMA,
        "completed": True,
        **hardware,
        "model_load_seconds": load_seconds,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "host": platform.node(),
        "input_pins": pins,
    })
    try:
        active_layers = sorted({layer for _split, layer in cell_buffers})
        decoded_by_layer = {
            layer: _prepare_layer_experts(
                args.checkpoint, args.trees, layer,
                [
                    expert
                    for key, buffer in cell_buffers.items()
                    if key[1] == layer
                    for row in buffer["allocations"]
                    for expert in row.expert_ids
                ],
                args.reconstruction_workers,
            )
            for layer in active_layers
        }
        device = model.get_input_embeddings().weight.device
        active_request_ids = sorted({
            request_id
            for buffer in cell_buffers.values()
            for request_id in buffer["by_request"]
        })
        for request_number, request_id in enumerate(active_request_ids, start=1):
            request = requests[request_id]
            request_keys = sorted(
                (
                    key for key, buffer in cell_buffers.items()
                    if request_id in buffer["by_request"]
                ),
                key=lambda key: (key[1], key[0]),
            )
            request_layers = sorted({layer for _split, layer in request_keys})
            encoded = _encoded_request(request, device)
            exact_prefix, prefix_seconds = advance_exact_prefix_native(
                model, encoded, request.position,
            )
            observer = MultiLayerCachedDecodeObserver(model, request_layers)
            try:
                baselines, repeat, baseline_seconds = _baseline_bundle(
                    observer, model, encoded, request.position,
                    exact_prefix, request_layers,
                )
                divisor = float(len(request_keys))
                for key in request_keys:
                    buffer = cell_buffers[key]
                    buffer["prefix_seconds_amortized"] += prefix_seconds / divisor
                    buffer["baseline_seconds_amortized"] += baseline_seconds / divisor
                    _execute_request_layer(
                        observer=observer,
                        model=model,
                        encoded=encoded,
                        exact_prefix=exact_prefix,
                        baseline=baselines[key[1]],
                        repeat=repeat,
                        request=request,
                        buffer=buffer,
                        decoded=decoded_by_layer[key[1]],
                        pins=pins,
                        q4_routed_max_abs_atol=Q4_ROUTED_MAX_ABS_ATOL,
                    )
            finally:
                observer.close()
            print(
                f"[nested outcomes request-major] "
                f"request={request_number}/{len(active_request_ids)} "
                f"layers={request_layers}",
                flush=True,
            )
            del baselines, repeat, exact_prefix, encoded
            gc.collect()
            torch.cuda.empty_cache()
        for key in sorted(cell_buffers, key=lambda value: (value[0], value[1])):
            _write_cell_buffer(
                buffer=cell_buffers[key], output=args.output, pins=pins,
            )
        del decoded_by_layer
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def finalize_phase(args: argparse.Namespace) -> None:
    (
        config, plan, allocations, requests, request_hashes,
        allocation_config_sha256, outcome_code_bundle,
    ) = _load_inputs(args)
    del requests
    checkpoint_hashes = {
        "checkpoint_config_sha256": str(
            load_json(args.config)["checkpoint_config_sha256"]
        ),
        "checkpoint_index_sha256": str(
            load_json(args.config)["checkpoint_index_sha256"]
        ),
    }
    pins = _input_pins(
        allocation_config_sha256=allocation_config_sha256,
        capture_hashes=request_hashes,
        checkpoint_hashes=checkpoint_hashes,
        selected_tree_sha256=str(load_json(args.config)["selected_tree_sha256"]),
        allocation_facts=outcome_input_facts(plan),
        execution_stack=dict(
            config["authenticated_capture_execution_stack"]
        ),
        outcome_code_bundle=outcome_code_bundle,
        q4_routed_max_abs_atol=Q4_ROUTED_MAX_ABS_ATOL,
    )
    quality_parts = []
    propagation_parts = []
    cache_parts = []
    zero_parts = []
    cell_facts = {}
    for split in args.splits:
        for layer in sorted({row.layer for row in allocations if row.split == split}):
            cell_allocations = tuple(
                row for row in allocations if row.split == split and row.layer == layer
            )
            paths = _cell_paths(args.output, split, layer)
            if not _completed_cell(
                paths, split=split, layer=layer, pins=pins,
                allocations=cell_allocations,
            ):
                raise RuntimeError(f"nested outcome cell is incomplete: {(split, layer)}")
            quality_parts.append(pd.read_parquet(paths["quality"]))
            propagation_parts.append(pd.read_parquet(paths["propagation"]))
            cache_parts.append(pd.read_parquet(paths["cache"]))
            zero_parts.append(pd.read_parquet(paths["zero"]))
            cell_facts[f"{split}/layer_{layer:02d}"] = load_json(paths["facts"])

    quality = pd.concat(quality_parts, ignore_index=True)
    propagation = pd.concat(propagation_parts, ignore_index=True)
    cache = pd.concat(cache_parts, ignore_index=True)
    zero = pd.concat(zero_parts, ignore_index=True)
    contrasts = route_mode_contrasts(quality)
    require_zero_dose_parity(zero)
    host_path = args.output / HOST_FACTS
    if not host_path.is_file():
        raise RuntimeError("nested outcome host facts are absent")
    host = load_json(host_path)
    required_hardware = config["hardware_execution_path"]
    if (
        host.get("schema") != OUTCOME_SCHEMA
        or host.get("completed") is not True
        or host.get("input_pins") != pins
        or str(required_hardware["required_gpu_name_substring"]) not in str(host.get("gpu"))
        or float(host.get("gpu_memory_gib", 0.0))
        < float(required_hardware["minimum_gpu_memory_gib"])
    ):
        raise RuntimeError("nested outcome host facts failed authentication")
    expected = {
        "quality_rows": len(allocations) * len(ROUTE_MODES),
        "propagation_rows": sum(
            (39 - row.layer) * len(ROUTE_MODES) for row in allocations
        ),
        "cache_rows": len(allocations) * len(ROUTE_MODES),
        "zero_rows": len({
            (row.split, row.layer, row.request_id) for row in allocations
        }) * len(ROUTE_MODES),
        "route_mode_contrast_rows": len(allocations),
    }
    observed = {
        "quality_rows": len(quality),
        "propagation_rows": len(propagation),
        "cache_rows": len(cache),
        "zero_rows": len(zero),
        "route_mode_contrast_rows": len(contrasts),
    }
    if observed != expected:
        raise RuntimeError(f"final nested outcome grid changed: {observed} != {expected}")
    execution_counts = {
        "physical_allocation_states": sum(
            int(facts["physical_allocation_states"])
            for facts in cell_facts.values()
        ),
        "physical_candidate_executions": sum(
            int(facts["physical_candidate_executions"])
            for facts in cell_facts.values()
        ),
        "logical_candidate_comparisons": sum(
            int(facts["logical_candidate_comparisons"])
            for facts in cell_facts.values()
        ),
        "zero_candidate_executions": sum(
            int(facts["zero_candidate_executions"])
            for facts in cell_facts.values()
        ),
    }
    if (
        execution_counts["logical_candidate_comparisons"] != observed["quality_rows"]
        or execution_counts["zero_candidate_executions"] != observed["zero_rows"]
        or execution_counts["physical_candidate_executions"]
        > execution_counts["logical_candidate_comparisons"]
    ):
        raise RuntimeError("final physical/logical execution accounting changed")
    for frame, name in (
        (quality, QUALITY_FILE),
        (propagation, PROPAGATION_FILE),
        (cache, CACHE_FILE),
        (zero, ZERO_FILE),
        (contrasts, CONTRAST_FILE),
    ):
        if frame.empty or frame.isnull().any().any():
            raise RuntimeError(f"final outcome table is empty or contains nulls: {name}")
        atomic_parquet(args.output / name, frame)
    outputs = {
        name: {
            "sha256": file_sha256(args.output / name),
            "bytes": (args.output / name).stat().st_size,
            "rows": len(frame),
        }
        for frame, name in (
            (quality, QUALITY_FILE),
            (propagation, PROPAGATION_FILE),
            (cache, CACHE_FILE),
            (zero, ZERO_FILE),
            (contrasts, CONTRAST_FILE),
        )
    }
    atomic_json(args.output / RUN_FACTS_FILE, {
        "schema": OUTCOME_SCHEMA,
        "completed": True,
        "input_pins": pins,
        "splits": list(args.splits),
        "layers": sorted({row.layer for row in allocations}),
        "counts": observed,
        "execution_counts": execution_counts,
        "outputs": outputs,
        "host_facts": {
            "file": HOST_FACTS,
            "sha256": file_sha256(host_path),
            "bytes": host_path.stat().st_size,
            "gpu": host["gpu"],
            "gpu_memory_gib": host["gpu_memory_gib"],
        },
        "cells": cell_facts,
        "terminal_metric_scope": "current_token_only",
        "one_exact_pretoken_cache_per_request_across_active_layers": True,
        "one_paired_native_baseline_per_request_across_active_layers": True,
        "exact_prefix_cache_cloned_per_physical_candidate": True,
        "identical_physical_candidates_fanned_out_to_logical_rows": True,
        "post_token_complete_cache_retained_and_audited": True,
        "zero_dose_all_cache_commits_exact": True,
        "every_zero_route_mode_physically_executed": True,
        "allocation_manifest_was_authenticated_before_access": True,
        "allocation_selected_from_terminal_outcomes": False,
        "no_allocation_decisions_in_outcome_runner": True,
        "experiment_b_started": False,
    })
    print(f"[finalized nested outcomes] {observed}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or finalize exact cached-decode outcomes from a sealed allocation manifest.",
    )
    parser.add_argument("--phase", choices=("run", "finalize"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--request-manifest-facts", type=Path)
    parser.add_argument("--allocation-manifest", type=Path, required=True)
    parser.add_argument("--allocation-seal", type=Path, required=True)
    parser.add_argument("--expected-allocation-sha256", required=True)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument(
        "--outcome-compatibility-protocol", type=Path, required=True,
    )
    parser.add_argument("--expected-compatibility-sha256", required=True)
    parser.add_argument(
        "--splits", choices=("calibration", "evaluation"), nargs="+",
        default=["evaluation"],
    )
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trees", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reconstruction-workers", type=int, default=8)
    args = parser.parse_args()
    args.splits = list(dict.fromkeys(args.splits))
    config = load_json(args.config)
    args.output = args.output or Path(str(config["output_root"])) / "outcomes"
    if args.phase == "run" and (args.checkpoint is None or args.trees is None):
        parser.error("--checkpoint and --trees are required for --phase run")
    if args.reconstruction_workers < 1 or args.reconstruction_workers > 64:
        parser.error("--reconstruction-workers must lie in [1,64]")
    return args


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parse_args()
    if args.phase == "run":
        run_phase(args)
    else:
        finalize_phase(args)


if __name__ == "__main__":
    main()
