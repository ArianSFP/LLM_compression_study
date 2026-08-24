#!/usr/bin/env python3
"""Legacy no-cache, full-sequence downstream-tail replay.

This implementation predates the exact-prefill decode product contract.  It
injects every admitted position in a request and scores the full sequence, so
it must not be cited or run as cached single-token decode evidence.

One layer is injected at a time through the same true pre-residual routed-MoE
replacement used by the same-host causal controls.  Each complete allocation
is propagated through native layers ``ell+1..39`` once in live routing and once
with fully frozen expert IDs and weights.  This is a three-request smoke study,
not an all-layer streamed-model quality curve.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.causal_control import FULLY_FROZEN, LIVE_ROUTES, PRE_RESIDUAL  # noqa: E402
from oracle_study.causal_replay import mean_squared_error, route_boundary_metrics  # noqa: E402
from oracle_study.d1_tail_quality import (  # noqa: E402
    STANDARD_POLICIES,
    add_live_minus_frozen,
    assemble_policy_delta_bank,
    calibrate_fixed_d1_policy,
    downstream_tail_metrics,
    index_oracle_cells,
    request_delta,
    request_policy_local_metrics,
    sha256,
    validate_tail_config,
)
from run_causal_impulse_replay import ForwardObserver, _tail_context  # noqa: E402
from run_d1_slice_oracle_pilot import _capture_path  # noqa: E402
from run_same_host_causal_controls import (  # noqa: E402
    SameHostObserver,
    _encoded_requests,
    _hidden_and_router,
    _load_model,
    _max_capture_difference,
    _run_candidate,
    _write_or_verify_baseline,
    _zero_reference,
    atomic_json,
    atomic_parquet,
)


SCHEMA = "pr13_d1_downstream_tail_kl_v1"
BASELINE_FACTS = "d1_tail_baseline_facts.json"
CALIBRATION = "d1_fixed_policy_calibration.parquet"
BANK_MANIFEST = "d1_policy_bank_manifest.json"
QUALITY = "d1_downstream_tail_quality.parquet"
PROPAGATION = "d1_downstream_tail_propagation.parquet"
RUN_FACTS = "d1_downstream_tail_run_facts.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _format_template(template: str, *, layer: int, rate: int) -> Path:
    return Path(str(template).format(layer=int(layer), rate=int(rate)))


def _cell_directories(config: Mapping[str, Any]) -> list[Path]:
    result = [Path(value) for value in config["existing_oracle_cell_dirs"]]
    template = str(config["matched_oracle_dir_template"])
    for rate in map(int, config["matched_runtime_metadata_page_caps"]):
        for layer in map(int, config["injection_layers"]):
            result.append(_format_template(template, layer=layer, rate=rate))
    return result


def _repair_directory(config: Mapping[str, Any], layer: int, rate: int) -> Path:
    matched = set(map(int, config["matched_runtime_metadata_page_caps"]))
    if int(rate) in matched:
        return _format_template(
            str(config["matched_repair_dir_template"]), layer=layer, rate=rate,
        )
    found = []
    for value in config["existing_repair_cell_dirs"]:
        directory = Path(str(value))
        facts = load_json(directory / "d1_exact_repair_facts.json")
        if (
            int(facts["layer"]) == int(layer)
            and int(facts["rate_pages_per_expert"]) == int(rate)
        ):
            found.append(directory)
    if len(found) != 1:
        raise RuntimeError(
            f"expected one exact repair directory for {(layer, rate)}; found {found}"
        )
    return found[0]


def _rates(config: Mapping[str, Any]) -> list[int]:
    return list(map(int, config["mechanism_page_caps"])) + list(map(
        int, config["matched_runtime_metadata_page_caps"],
    ))


def _prepare_banks(
    config: Mapping[str, Any],
) -> tuple[dict[tuple[int, int], Any], dict[int, str], pd.DataFrame, dict[tuple[int, int], Any]]:
    cells = index_oracle_cells(_cell_directories(config))
    expected = {
        (layer, rate)
        for layer in map(int, config["injection_layers"])
        for rate in _rates(config)
    }
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        unexpected = sorted(set(cells) - expected)
        raise RuntimeError(f"tail oracle cell grid changed: missing={missing}, unexpected={unexpected}")
    calibration = config["fixed_d1_calibration"]
    fixed, evidence = calibrate_fixed_d1_policy(
        cells,
        calibration["calibration_layers"],
        _rates(config),
    )
    banks = {}
    for key, cell in sorted(cells.items()):
        layer, rate = key
        banks[key] = assemble_policy_delta_bank(
            cell,
            _repair_directory(config, layer, rate),
            fixed[rate],
        )
    return cells, fixed, evidence, banks


def _bank_manifest(
    config: Mapping[str, Any],
    cells: Mapping[tuple[int, int], Any],
    fixed: Mapping[int, str],
    banks: Mapping[tuple[int, int], Any],
    config_path: Path,
) -> dict[str, Any]:
    implementation_files = [
        EXPERIMENT / "src/oracle_study/d1_tail_quality.py",
        EXPERIMENT / "src/oracle_study/causal_control.py",
        EXPERIMENT / "src/oracle_study/causal_replay.py",
        EXPERIMENT / "scripts/run_d1_full_sequence_tail_kl_legacy.py",
        EXPERIMENT / "scripts/analyze_d1_full_sequence_tail_kl_legacy.py",
        EXPERIMENT / "scripts/run_causal_impulse_replay.py",
        EXPERIMENT / "scripts/run_same_host_causal_controls.py",
    ]
    records = {}
    for (layer, rate), bank in sorted(banks.items()):
        cell = cells[(layer, rate)]
        repair = _repair_directory(config, layer, rate)
        records[f"layer_{layer:02d}_rate_{rate}"] = {
            "layer": layer,
            "rate_pages_per_expert": rate,
            "oracle_directory": str(cell.directory),
            "oracle_facts_sha256": sha256(cell.directory / "d1_layer_facts.json"),
            "oracle_metrics_sha256": sha256(
                cell.directory / "d1_exact_route_metrics.parquet"
            ),
            "oracle_deltas_sha256": sha256(cell.directory / "d1_selected_deltas.npz"),
            "repair_directory": str(repair),
            "repair_facts_sha256": sha256(repair / "d1_exact_repair_facts.json"),
            "repair_metrics_sha256": sha256(
                repair / "d1_exact_repair_metrics.parquet"
            ),
            "repair_deltas_sha256": sha256(repair / "d1_exact_repair_delta.npz"),
            "policies": list(bank.deltas),
            "policy_provenance": bank.provenance,
        }
    return {
        "completed": True,
        "schema": SCHEMA,
        "config_sha256": sha256(config_path),
        "base_d1_expansion_config_sha256": str(
            config["base_d1_expansion_config_sha256"]
        ),
        "implementation_files": {
            str(path.relative_to(EXPERIMENT)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in implementation_files
        },
        "fixed_d1_source_policy_by_rate": {
            str(rate): policy for rate, policy in sorted(fixed.items())
        },
        "cells": records,
        "test_rows_admitted_or_used": False,
    }


def validate_phase(args: argparse.Namespace) -> None:
    cells, fixed, evidence, banks = _prepare_banks(args.config_data)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_parquet(args.output / CALIBRATION, evidence)
    atomic_json(
        args.output / BANK_MANIFEST,
        _bank_manifest(args.config_data, cells, fixed, banks, args.config),
    )
    print(
        "[validated] fixed policies "
        + ", ".join(f"rate {rate}: {policy}" for rate, policy in sorted(fixed.items())),
        flush=True,
    )


def _source_capture_parity(
    source_directory: Path,
    captures: Mapping[str, Mapping[str, Any]],
    layers: list[int],
) -> dict[str, Any]:
    rows = {}
    for request_id, current in captures.items():
        path = _capture_path(source_directory, request_id)
        maxima = {}
        with np.load(path, allow_pickle=False) as source:
            fields = {
                "input_ids": np.asarray(current["input_ids"]),
                "attention_mask": np.asarray(current["attention_mask"]),
            }
            for layer in layers:
                fields[f"residual_layer_{layer:02d}"] = np.asarray(
                    current["residual"][layer], np.float32,
                )
                fields[f"x_layer_{layer:02d}"] = np.asarray(
                    current["x"][layer], np.float32,
                )
                fields[f"routed_layer_{layer:02d}"] = np.asarray(
                    current["routed"][layer], np.float32,
                )
            for name, value in fields.items():
                reference = np.asarray(source[name])
                if reference.shape != value.shape:
                    raise RuntimeError(f"source capture shape changed: {request_id}/{name}")
                difference = float(np.max(np.abs(
                    reference.astype(np.float64) - value.astype(np.float64)
                ), initial=0.0))
                maxima[name] = difference
                if not np.array_equal(reference, value):
                    raise RuntimeError(
                        f"source capture is not exact on this host: {request_id}/{name} "
                        f"max_abs={difference:.9g}",
                    )
        rows[request_id] = {
            "source_file": str(path),
            "source_sha256": sha256(path),
            "field_max_abs": maxima,
        }
    return rows


def _runtime_provenance() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "cpu_count": os.cpu_count(),
    }


def _capture_baselines(
    args: argparse.Namespace,
    model: Any,
    encoded: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], float]:
    layers = list(map(int, args.config_data["injection_layers"]))
    observer = SameHostObserver(model, layers)
    captures = {}
    repeat = {}
    elapsed_total = 0.0
    try:
        for request_id in map(str, args.config_data["validation_request_ids"]):
            first, elapsed = observer.run(model, encoded[request_id])
            elapsed_total += elapsed
            second, elapsed = observer.run(model, encoded[request_id])
            elapsed_total += elapsed
            difference = _max_capture_difference(first, second)
            if any(float(value) != 0.0 for value in difference.values()):
                raise RuntimeError("same-process repeated Q4 baseline is not bit-identical")
            captures[request_id] = first
            repeat[request_id] = difference
    finally:
        observer.close()
    parity = _source_capture_parity(
        Path(args.config_data["source_capture_dir"]), captures, layers,
    )
    _write_or_verify_baseline(args.output, captures, layers)
    return captures, {"repeat_max_abs": repeat, "source_capture_parity": parity}, elapsed_total


def _cell_paths(output: Path, layer: int, rate: int) -> dict[str, Path]:
    root = output / "cells"
    stem = f"layer_{int(layer):02d}_rate_{int(rate)}"
    return {
        "quality": root / f"d1_tail_quality_{stem}.parquet",
        "propagation": root / f"d1_tail_propagation_{stem}.parquet",
        "sidecar": root / f"d1_tail_{stem}.json",
        "failure": root / f"d1_tail_{stem}.failure.json",
    }


def _propagation_rows(
    *,
    layer: int,
    request_id: str,
    rate: int,
    policy: str,
    route_mode: str,
    reference_hidden: Mapping[int, np.ndarray],
    candidate_hidden: Mapping[int, np.ndarray],
    reference_router: Mapping[int, np.ndarray],
    candidate_router: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for observation in range(int(layer) + 1, 40):
        boundary = route_boundary_metrics(
            np.asarray(reference_router[observation], np.float32),
            np.asarray(candidate_router[observation], np.float32),
        )
        rows.append({
            "schema": SCHEMA,
            "injection_layer": int(layer),
            "observation_layer": observation,
            "layer_distance": observation - int(layer),
            "request_id": str(request_id),
            "rate_pages_per_expert": int(rate),
            "policy": str(policy),
            "route_mode": str(route_mode),
            "hidden_mse": mean_squared_error(
                np.asarray(reference_hidden[observation], np.float32),
                np.asarray(candidate_hidden[observation], np.float32),
            ),
            **boundary,
        })
    return rows


def _run_cell(
    args: argparse.Namespace,
    model: Any,
    observer: ForwardObserver,
    captures: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    bank: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    layer = int(bank.layer)
    rate = int(bank.rate)
    config = args.config_data
    accounting = config["page_cap_accounting"][str(rate)]
    quality_rows = []
    propagation_rows = []
    zero_rows = []
    candidate_seconds = 0.0
    for request_id in map(str, config["validation_request_ids"]):
        baseline = captures[request_id]
        references = {}
        for route_mode in config["route_modes"]:
            hidden, router, logits, facts = _zero_reference(
                observer, model, baseline, contexts[request_id], layer, str(route_mode),
            )
            if any(
                float(facts[name]) > float(config["zero_dose_hidden_router_and_logits_max_abs_atol"])
                for name in ("hidden_max_abs", "router_max_abs", "logits_max_abs")
            ):
                raise RuntimeError("zero-dose tail did not exactly reproduce Q4")
            references[str(route_mode)] = (hidden, router, logits)
            zero_rows.append({"request_id": request_id, "route_mode": route_mode, **facts})

        sequence_length = int(len(baseline["input_ids"]))
        for policy in STANDARD_POLICIES:
            delta, positions = request_delta(
                bank, policy, request_id, sequence_length,
            )
            local_metrics = request_policy_local_metrics(
                bank, policy, request_id,
            )
            for route_mode in config["route_modes"]:
                reference_hidden, reference_router, reference_logits = references[str(route_mode)]
                start, tail_hidden, tail_router, candidate_logits, injection, elapsed = _run_candidate(
                    observer,
                    model,
                    baseline,
                    contexts[request_id],
                    layer,
                    delta,
                    1.0,
                    PRE_RESIDUAL,
                    str(route_mode),
                )
                candidate_seconds += elapsed
                candidate_hidden, candidate_router = _hidden_and_router(
                    baseline, layer, start, tail_hidden, tail_router,
                )
                metrics = downstream_tail_metrics(
                    reference_logits=np.asarray(reference_logits, np.float32),
                    candidate_logits=np.asarray(candidate_logits, np.float32),
                    input_ids=np.asarray(baseline["input_ids"], np.int64),
                    reference_final_hidden=np.asarray(reference_hidden[39], np.float32),
                    candidate_final_hidden=np.asarray(candidate_hidden[39], np.float32),
                    reference_router=reference_router,
                    candidate_router=candidate_router,
                    injection_layer=layer,
                    admitted_positions=positions,
                )
                realized_layer_mse = mean_squared_error(
                    np.asarray(reference_hidden[layer], np.float32),
                    np.asarray(candidate_hidden[layer], np.float32),
                )
                realized_injected_layer_mse = mean_squared_error(
                    np.asarray(reference_hidden[layer], np.float32)[positions],
                    np.asarray(candidate_hidden[layer], np.float32)[positions],
                )
                quality_rows.append({
                    "schema": SCHEMA,
                    "injection_layer": layer,
                    "request_id": request_id,
                    "rate_pages_per_expert": rate,
                    "traffic_equivalent_pr13_pages": int(
                        accounting["traffic_equivalent_pr13_pages"]
                    ),
                    "charged_bpw": float(accounting["charged_bpw"]),
                    "point_type": str(accounting["point_type"]),
                    "policy": policy,
                    "route_mode": str(route_mode),
                    "sequence_tokens": sequence_length,
                    "label_tokens": sequence_length - 1,
                    "injected_positions": len(positions),
                    "realized_layer_output_mse": realized_layer_mse,
                    "realized_injected_layer_output_mse": realized_injected_layer_mse,
                    "tail_seconds": elapsed,
                    **local_metrics,
                    **injection,
                    **metrics,
                })
                propagation_rows.extend(_propagation_rows(
                    layer=layer,
                    request_id=request_id,
                    rate=rate,
                    policy=policy,
                    route_mode=str(route_mode),
                    reference_hidden=reference_hidden,
                    candidate_hidden=candidate_hidden,
                    reference_router=reference_router,
                    candidate_router=candidate_router,
                ))
                print(
                    f"[tail] layer={layer} rate={rate} request={request_id} "
                    f"policy={policy} mode={route_mode} KL={metrics['logit_kl']:.8g}",
                    flush=True,
                )
    return quality_rows, propagation_rows, {
        "zero_rows": zero_rows,
        "candidate_tail_seconds": candidate_seconds,
        "policy_provenance": bank.provenance,
    }


def run_phase(args: argparse.Namespace) -> None:
    validate_phase(args)
    config = args.config_data
    if not torch.cuda.is_available():
        raise RuntimeError("exact downstream-tail replay requires CUDA")
    hardware = config["hardware_execution_path"]
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if str(hardware["required_gpu_name_substring"]) not in gpu_name:
        raise RuntimeError(f"exact-tail GPU identity changed: {gpu_name}")
    if gpu_memory < float(hardware["minimum_gpu_memory_gib"]):
        raise RuntimeError(
            f"exact-tail GPU memory is insufficient: {gpu_memory:.3f} GiB"
        )
    cells, fixed, _, banks = _prepare_banks(config)
    model, tokenizer, model_load_seconds = _load_model(args.checkpoint)
    encoded = _encoded_requests(tokenizer, model, config)
    captures, baseline_checks, capture_seconds = _capture_baselines(args, model, encoded)
    baseline_payload = {
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "model_load_seconds": model_load_seconds,
        "capture_seconds": capture_seconds,
        **baseline_checks,
        "runtime_provenance": _runtime_provenance(),
        "test_rows_admitted_or_used": False,
    }
    atomic_json(args.output / BASELINE_FACTS, baseline_payload)

    device = model.get_input_embeddings().weight.device
    contexts = {}
    for request_id in map(str, config["validation_request_ids"]):
        seed = torch.as_tensor(
            captures[request_id]["hidden"][0], device=device, dtype=torch.bfloat16,
        ).unsqueeze(0)
        contexts[request_id] = _tail_context(model, encoded[request_id], seed)
    observer = ForwardObserver(model)
    try:
        for key in sorted(banks):
            layer, rate = key
            paths = _cell_paths(args.output, layer, rate)
            if paths["failure"].exists():
                raise RuntimeError(f"prior D1 tail failure must be reviewed: {paths['failure']}")
            if any(paths[name].exists() for name in ("quality", "propagation", "sidecar")):
                if not all(paths[name].is_file() for name in ("quality", "propagation", "sidecar")):
                    raise RuntimeError("partial D1 tail cell exists")
                sidecar = load_json(paths["sidecar"])
                if (
                    sidecar["quality_sha256"] != sha256(paths["quality"])
                    or sidecar["propagation_sha256"] != sha256(paths["propagation"])
                ):
                    raise RuntimeError("resumed D1 tail cell hash changed")
                continue
            started = time.perf_counter()
            try:
                quality, propagation, facts = _run_cell(
                    args, model, observer, captures, contexts, banks[key],
                )
                atomic_parquet(paths["quality"], quality)
                atomic_parquet(paths["propagation"], propagation)
                sidecar = {
                    "completed": True,
                    "schema": SCHEMA,
                    "run_id": config["run_id"],
                    "layer": layer,
                    "rate_pages_per_expert": rate,
                    "fixed_d1_source_policy": fixed[rate],
                    "quality_rows": len(quality),
                    "propagation_rows": len(propagation),
                    "wall_seconds": time.perf_counter() - started,
                    "quality_sha256": sha256(paths["quality"]),
                    "propagation_sha256": sha256(paths["propagation"]),
                    "oracle_facts_sha256": sha256(
                        cells[key].directory / "d1_layer_facts.json"
                    ),
                    "test_rows_admitted_or_used": False,
                    **facts,
                }
                atomic_json(paths["sidecar"], sidecar)
            except Exception as error:
                atomic_json(paths["failure"], {
                    "completed": False,
                    "schema": SCHEMA,
                    "run_id": config["run_id"],
                    "layer": layer,
                    "rate_pages_per_expert": rate,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                })
                raise
    finally:
        observer.close()
        del model
        torch.cuda.empty_cache()
        gc.collect()


def finalize_phase(args: argparse.Namespace) -> None:
    config = args.config_data
    quality_parts = []
    propagation_parts = []
    cells = {}
    zero_dose_rows = 0
    for layer in map(int, config["injection_layers"]):
        for rate in _rates(config):
            paths = _cell_paths(args.output, layer, rate)
            if paths["failure"].exists():
                raise RuntimeError(f"D1 tail failure exists: layer={layer}, rate={rate}")
            if not all(paths[name].is_file() for name in ("quality", "propagation", "sidecar")):
                raise RuntimeError(f"D1 tail cell is incomplete: layer={layer}, rate={rate}")
            sidecar = load_json(paths["sidecar"])
            if (
                sidecar["quality_sha256"] != sha256(paths["quality"])
                or sidecar["propagation_sha256"] != sha256(paths["propagation"])
            ):
                raise RuntimeError("D1 tail cell changed before finalization")
            expected_cell_quality = (
                len(config["validation_request_ids"])
                * len(config["policies"])
                * len(config["route_modes"])
            )
            expected_cell_propagation = expected_cell_quality * (39 - int(layer))
            if int(sidecar["quality_rows"]) != expected_cell_quality:
                raise RuntimeError("D1 tail cell quality row count changed")
            if int(sidecar["propagation_rows"]) != expected_cell_propagation:
                raise RuntimeError("D1 tail cell propagation row count changed")
            if len(sidecar["zero_rows"]) != (
                len(config["validation_request_ids"]) * len(config["route_modes"])
            ):
                raise RuntimeError("D1 tail zero-dose row count changed")
            zero_dose_rows += len(sidecar["zero_rows"])
            quality_parts.append(pd.read_parquet(paths["quality"]))
            propagation_parts.append(pd.read_parquet(paths["propagation"]))
            cells[f"layer_{layer:02d}_rate_{rate}"] = sidecar
    quality = add_live_minus_frozen(pd.concat(quality_parts, ignore_index=True))
    propagation = pd.concat(propagation_parts, ignore_index=True)
    expected_quality = (
        len(config["injection_layers"])
        * len(_rates(config))
        * len(config["validation_request_ids"])
        * len(config["policies"])
        * len(config["route_modes"])
    )
    expected_propagation = (
        sum(39 - int(layer) for layer in config["injection_layers"])
        * len(_rates(config))
        * len(config["validation_request_ids"])
        * len(config["policies"])
        * len(config["route_modes"])
    )
    if len(quality) != expected_quality or len(propagation) != expected_propagation:
        raise RuntimeError("final D1 tail row grid changed")
    numeric = quality.select_dtypes(include=[np.number]).drop(
        columns=["first_route_membership_change_layer"], errors="ignore",
    ).to_numpy(np.float64)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("non-finite D1 tail quality result")
    if not np.all(np.isfinite(propagation.select_dtypes(include=[np.number]))):
        raise RuntimeError("non-finite D1 tail propagation result")
    atomic_parquet(args.output / QUALITY, quality)
    atomic_parquet(args.output / PROPAGATION, propagation)
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "layers": list(map(int, config["injection_layers"])),
        "rates": _rates(config),
        "policies": list(STANDARD_POLICIES),
        "route_modes": list(map(str, config["route_modes"])),
        "requests": list(map(str, config["validation_request_ids"])),
        "quality_rows": len(quality),
        "propagation_rows": len(propagation),
        "zero_dose_rows": zero_dose_rows,
        "quality_sha256": sha256(args.output / QUALITY),
        "propagation_sha256": sha256(args.output / PROPAGATION),
        "config_sha256": sha256(args.config),
        "bank_manifest_sha256": sha256(args.output / BANK_MANIFEST),
        "calibration_sha256": sha256(args.output / CALIBRATION),
        "baseline_facts_sha256": sha256(args.output / BASELINE_FACTS),
        "cells": cells,
        "smoke_only": True,
        "minimum_requests_before_quality_claim": int(
            config["minimum_requests_before_quality_claim"]
        ),
        "test_rows_admitted_or_used": False,
        "scientific_boundary": config["scientific_boundary"],
    }
    atomic_json(args.output / RUN_FACTS, facts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validate", "run", "finalize"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.config_data = load_json(args.config)
    validate_tail_config(args.config_data)
    expansion_config = EXPERIMENT / str(
        args.config_data["base_d1_expansion_config"]
    )
    if sha256(expansion_config) != str(
        args.config_data["base_d1_expansion_config_sha256"]
    ):
        raise RuntimeError("immutable D1 expansion config hash changed")
    if args.output is None:
        args.output = Path(args.config_data["output_root"])
    if args.phase == "run" and args.checkpoint is None:
        parser.error("--checkpoint is required for --phase run")
    if args.checkpoint is not None:
        index = args.checkpoint / "model.safetensors.index.json"
        if sha256(index) != args.config_data["checkpoint_index_sha256"]:
            raise RuntimeError("checkpoint index hash changed")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "validate":
        validate_phase(args)
    elif args.phase == "run":
        run_phase(args)
    else:
        finalize_phase(args)


if __name__ == "__main__":
    main()
