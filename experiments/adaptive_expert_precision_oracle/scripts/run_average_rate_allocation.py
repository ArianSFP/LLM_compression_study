#!/usr/bin/env python3
"""Fit all-expert interaction sidecars and evaluate grouped average rate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.average_rate_allocator import (  # noqa: E402
    AllocationTrace,
    RateOption,
    allocation_aware_rate_frontiers,
    bounded_group_exchange_allocate,
    exact_group_option_allocate,
    global_group_dual_bound,
    lagrangian_rate_frontier,
    multiple_choice_allocate,
    qmetric_features,
    randomized_joint_metric_factor,
)
from oracle_study.interaction_field import (  # noqa: E402
    EncodedInteractionFactor,
    JointInteractionFactor,
    encode_interaction_factor,
    factor_exact_proxy_plus_tail,
    make_encoded_factor_self_safe,
    make_factor_self_safe,
)
from oracle_study.mxfp4_embed import load_compressed_mxfp4_expert  # noqa: E402
from oracle_study.neuron_selector import unit_score_metadata  # noqa: E402
from oracle_study.split_interaction_field import (  # noqa: E402
    SPLIT_STATE_PAGE_COSTS,
    build_split_interaction_field,
    split_projection_responses,
    split_state_output,
)
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_set_utility_distillation as prior  # noqa: E402
import run_sparse_streaming_study as base  # noqa: E402


FIT_FACTS = "average_rate_fit_facts.json"
FIT_MANIFEST = "average_rate_factor_manifest.json"
FIT_LAYER = "average_rate_factor_layer_{layer}.npz"
FIT_SIDECAR = "average_rate_factor_layer_{layer}.json"
RUN_FACTS = "average_rate_run_facts.json"
GROUP_FRONTIER = "average_rate_group_frontier.parquet"
EXPERT_ALLOCATION = "average_rate_expert_allocation.parquet"
ACCOUNTING = "average_rate_accounting.json"
GROUP_IDENTITY = (
    "capture_source", "evaluation_split", "request_id", "position", "layer",
)
PROJECTIONS = ("gate", "up", "down")
UNITS = 512
EXPERT_WEIGHTS = 3_145_728
PAGE_BYTES = 512
_EVAL_CONTEXT: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    return prior.sha256(path)


def _atomic_json(path: Path, payload: Any) -> None:
    prior.atomic_json(path, payload)


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(list(rows)).to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _cpu_capacity() -> dict[str, Any]:
    quota, period = (
        Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    )
    quota_cores = None
    raw = "unavailable"
    if Path("/sys/fs/cgroup/cpu.max").exists():
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        values = raw.split()
        if len(values) == 2 and values[0] != "max":
            quota_cores = float(values[0]) / float(values[1])
    elif quota.exists() and period.exists():
        raw = f"{quota.read_text().strip()} {period.read_text().strip()}"
        value, width = map(int, raw.split())
        if value > 0:
            quota_cores = value / width
    return {
        "cpu_quota_raw": raw,
        "quota_cores": quota_cores,
        "affinity_logical_cpus": len(os.sched_getaffinity(0)),
        "os_cpu_count": os.cpu_count(),
    }


def _dependency_hashes() -> dict[str, str]:
    paths = {
        "runner_sha256": Path(__file__).resolve(),
        "allocator_core_sha256": EXPERIMENT / "src/oracle_study/average_rate_allocator.py",
        "split_core_sha256": EXPERIMENT / "src/oracle_study/split_interaction_field.py",
        "interaction_core_sha256": EXPERIMENT / "src/oracle_study/interaction_field.py",
        "selector_core_sha256": EXPERIMENT / "src/oracle_study/neuron_selector.py",
        "mxfp4_core_sha256": EXPERIMENT / "src/oracle_study/mxfp4_embed.py",
        "set_utility_runner_sha256": EXPERIMENT / "scripts/run_set_utility_distillation.py",
        "sparse_runner_sha256": EXPERIMENT / "scripts/run_sparse_streaming_study.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _validate_contract(config: Mapping[str, Any]) -> None:
    if config.get("base_pr12_commit") != "32601905f45c2e7caf402fd0a6d297d3fbd2a6d4":
        raise RuntimeError("average-rate study is not stacked on frozen PR #12")
    if config.get("fit_split") != "train" or config.get("selection_split") != "validation":
        raise RuntimeError("split contract changed")
    if config.get("no_test_rows_admitted") is not True:
        raise RuntimeError("test-row admission must remain forbidden")
    if list(map(int, config.get("layers", []))) != [0, 4, 20, 39]:
        raise RuntimeError("layer grid changed")
    if int(config.get("experts_per_group", -1)) != 8:
        raise RuntimeError("top-8 group width changed")
    if int(config.get("experts_per_layer", -1)) != 256:
        raise RuntimeError("all-expert fit scope changed")
    if int(config.get("expected_validation_groups", -1)) != 128:
        raise RuntimeError("validation group count changed")
    if list(map(int, config.get("mean_correction_page_budgets", []))) != [384, 576, 749, 768]:
        raise RuntimeError("mean rate grid changed")
    if list(map(int, config.get("primary_burst_caps_pages", []))) != [768, 1152, 1536]:
        raise RuntimeError("burst-cap grid changed")
    if int(config.get("frontier_maximum_pages", -1)) != 1536:
        raise RuntimeError("frontier endpoint changed")
    if int(config.get("evaluation_workers", -1)) != 24:
        raise RuntimeError("worker count changed")
    if config.get("worker_start_method") != "fork" or int(config.get("worker_blas_threads", -1)) != 1:
        raise RuntimeError("worker execution contract changed")
    if int(config.get("primary_factor_payload_bytes", -1)) != 6148:
        raise RuntimeError("INT4 factor payload changed")
    if int(config.get("compute_control_factor_payload_bytes", -1)) != 16384:
        raise RuntimeError("rank-4 factor payload changed")
    if int(config.get("abc_metadata_bytes_per_expert", -1)) != 3084:
        raise RuntimeError("A/B/C payload changed")
    if int(config.get("factor_fit_rank", -1)) != 8:
        raise RuntimeError("primary factor rank changed")
    expected_factors = [
        ("matrix_free_joint_rank8_int4_per_row_hadamard", 8, 6148, 749, "full_primary"),
        ("exact_proxy_rank4_fp32", 4, 16384, 729, "strict_control"),
        ("exact_proxy_rank4_fp16", 4, 8196, 745, "strict_control"),
        ("exact_proxy_rank4_int8_per_row", 4, 6148, 749, "strict_control"),
        ("exact_proxy_rank4_int4_per_row_hadamard", 4, 4100, 753, "strict_control"),
        ("exact_proxy_rank4_int8_plus_euclidean_tail4_int4", 8, 10244, 741, "strict_control"),
    ]
    observed_factors = [
        (
            str(record.get("factor_id")), int(record.get("rank", -1)),
            int(record.get("factor_payload_bytes", -1)),
            int(record.get("all_in_mean_pages", -1)),
            str(record.get("evaluation_grid")),
        )
        for record in config.get("factor_configs", [])
    ]
    if observed_factors != expected_factors:
        raise RuntimeError("factor control grid changed")
    strict_bytes = int(config["strict_all_in_total_bytes_per_expert"])
    abc_bytes = int(config["abc_metadata_bytes_per_expert"])
    for _, _, payload_bytes, pages, _ in expected_factors:
        if (strict_bytes - abc_bytes - payload_bytes) // PAGE_BYTES != pages:
            raise RuntimeError("strict all-in factor page arithmetic changed")
    if list(config.get("quantized_control_factor_ids", [])) != [
        record[0] for record in expected_factors[2:]
    ]:
        raise RuntimeError("quantized rank-4 factor grid changed")
    if int(config.get("mixed_euclidean_tail_rank", -1)) != 4:
        raise RuntimeError("mixed Euclidean tail rank changed")
    if int(config.get("price_local_max_passes", -1)) != 0:
        raise RuntimeError("price-path repair contract changed")
    targets = list(map(int, config.get("frontier_target_pages", [])))
    if targets != [0, 384, 576, 729, 749, 768, 1152, 1536]:
        raise RuntimeError("frontier hard-anchor grid changed")
    prices = list(map(float, config.get("frontier_price_ratios", [])))
    if not prices or prices != sorted(prices, reverse=True) or prices[-1] != 0.0:
        raise RuntimeError("frontier page-price path changed")
    if int(config.get("column_generation_max_rounds", -1)) != 3:
        raise RuntimeError("column-generation round cap changed")
    if list(map(float, config.get("column_generation_price_multipliers", []))) != [0.5, 2.0]:
        raise RuntimeError("adaptive price insertion changed")
    if int(config.get("column_generation_price_local_max_passes", -1)) != 0:
        raise RuntimeError("adaptive price local-repair contract changed")
    if (
        config.get("global_bound_factor_id") != config.get("primary_factor_id")
        or int(config.get("global_bound_mean_pages", -1)) != 749
        or int(config.get("global_bound_burst_cap_pages", -1)) != 1536
        or int(config.get("global_bound_max_iterations", -1)) != 128
        or float(config.get("global_bound_relative_tolerance", -1.0)) != 1e-6
        or list(map(int, config.get("global_exchange_sizes", []))) != [3, 4]
        or int(config.get("global_exchange_shortlist", -1)) != 8
        or int(config.get("global_exchange_max_passes", -1)) != 2
    ):
        raise RuntimeError("global group-bound contract changed")



def _expected_policy_grid(
    config: Mapping[str, Any],
) -> set[tuple[str, str, int, int]]:
    primary = str(config["primary_factor_id"])
    result: set[tuple[str, str, int, int]] = set()
    for mean in map(int, config["mean_correction_page_budgets"]):
        result.add((primary, "uniform_per_expert", mean, mean))
        for burst in map(int, config["primary_burst_caps_pages"]):
            result.add((
                primary, "pooled_router_square_compressed", mean, burst,
            ))
            result.add((
                primary, "pooled_router_square_column_generated", mean, burst,
            ))
        result.add((
            primary, "pooled_equal_weight_compressed", mean, 1536,
        ))
        result.add((
            primary, "pooled_exact_combined_moe_oracle", mean, 1536,
        ))
        result.add((
            primary,
            "pooled_exact_combined_moe_column_generated_local",
            mean, 1536,
        ))
        if mean == int(config["global_bound_mean_pages"]):
            result.add((
                primary, "pooled_exact_combined_moe_global_bound",
                mean, int(config["global_bound_burst_cap_pages"]),
            ))
    for record in config["factor_configs"]:
        factor_id = str(record["factor_id"])
        if factor_id == primary:
            continue
        mean = int(record["all_in_mean_pages"])
        result |= {
            (factor_id, "uniform_per_expert", mean, mean),
            (factor_id, "pooled_router_square_compressed", mean, 1536),
            (
                factor_id, "pooled_router_square_column_generated",
                mean, 1536,
            ),
            (factor_id, "pooled_exact_combined_moe_oracle", mean, 1536),
            (
                factor_id,
                "pooled_exact_combined_moe_column_generated_local",
                mean, 1536,
            ),
        }
    return result

def _verify_base(
    config: Mapping[str, Any], pr12_config: Path, pr12_dir: Path,
) -> dict[str, str]:
    paths = {
        "base_pr12_config_sha256": pr12_config,
        "base_pr12_runner_sha256": EXPERIMENT / "scripts/run_split_interaction_field_study.py",
        "base_pr12_run_facts_sha256": pr12_dir / "run_facts.json",
        "base_pr12_frontier_sha256": pr12_dir / "split_interaction_field_frontier.parquet",
    }
    result = {}
    for name, path in paths.items():
        observed = _sha256(path)
        if observed != str(config[name]):
            raise RuntimeError(f"frozen PR #12 dependency changed: {path}")
        result[name] = observed
    facts = json.loads((pr12_dir / "run_facts.json").read_text())
    if facts.get("split_core_sha256") != config["base_pr12_core_sha256"]:
        raise RuntimeError("frozen PR #12 core hash changed in its immutable facts")
    return result


def _locked_inputs(
    config: Mapping[str, Any], pr10_config: Path, capture: Path,
    checkpoint: Path, trees: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    if _sha256(pr10_config) != str(config["pr10_config_sha256"]):
        raise RuntimeError("frozen PR #10 config changed")
    old = json.loads(pr10_config.read_text())
    return base.verify_locked_inputs(
        old, pr10_config, capture, checkpoint, trees, "exact_checkpoint",
    )


def _thread_contract(config: Mapping[str, Any]) -> dict[str, str | None]:
    values = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if set(values.values()) != {str(config["worker_blas_threads"])}:
        raise RuntimeError("BLAS thread environment is not frozen")
    capacity = _cpu_capacity()
    if capacity["quota_cores"] is not None and float(capacity["quota_cores"]) < int(
        config["evaluation_workers"]
    ):
        raise RuntimeError("CPU quota is below the frozen worker count")
    return values


def _stored_downward_scale(value: float) -> np.float32:
    target = float(value)
    stored = np.float32(target)
    if float(stored) > target:
        stored = np.nextafter(stored, np.float32(0.0))
    if not np.isfinite(stored) or stored < 0:
        raise RuntimeError("factor shrink is not representable")
    return stored


def _fit_probe_error(
    down2: np.ndarray, down4: np.ndarray, proxy: np.ndarray, beta: float,
    factor: JointInteractionFactor, seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    probes = rng.choice(
        np.asarray([-1.0, 1.0], np.float32), size=(2 * UNITS, 8),
    )
    joint = np.concatenate((down4, down2), axis=1).astype(np.float64)
    output = joint @ probes
    exact = np.einsum("ij,ij->j", output, output, optimize=True)
    if float(beta) != 0.0:
        projected = output.T @ np.asarray(proxy, np.float64)
        exact += float(beta) * np.einsum("ij,ij->i", projected, projected, optimize=True)
    rows = np.concatenate((factor.l4, factor.l2), axis=0).astype(np.float64)
    latent = rows.T @ probes
    approximate = np.einsum("ij,ij->j", latent, latent, optimize=True)
    relative = np.abs(approximate - exact) / np.maximum(exact, 1e-30)
    return float(np.median(relative)), float(np.quantile(relative, .9))


def _fit_layer(
    config: Mapping[str, Any], checkpoint: Path, index: Mapping[str, str],
    tree: Any, layer: int, proxy: np.ndarray, beta: float, device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    primary_codes, primary_scales, primary_global = [], [], []
    rank4_l4, rank4_l2 = [], []
    fp16_l4, fp16_l2, fp16_global = [], [], []
    int8_codes, int8_scales, int8_global = [], [], []
    int4_codes, int4_scales, int4_global = [], [], []
    mixed_tail_codes, mixed_tail_scales = [], []
    mixed_proxy_codes, mixed_proxy_scales, mixed_global = [], [], []
    primary_shrink, fp16_shrink, mixed_shrink = [], [], []
    probe_median, probe_p90 = [], []
    decode_seconds = factor_seconds = 0.0
    empty_proxy = np.empty((proxy.shape[0], 0), np.float32)
    for expert in range(int(config["experts_per_layer"])):
        started = time.perf_counter()
        leaves = load_compressed_mxfp4_expert(
            checkpoint, dict(index), layer, expert, "down",
        )
        down2 = tree.decode(leaves, 2)
        down4 = tree.decode(leaves, 4)
        decode_seconds += time.perf_counter() - started
        abc = unit_score_metadata(down2, down4, proxy=proxy, beta=beta)
        started = time.perf_counter()

        primary = randomized_joint_metric_factor(
            down2, down4, proxy, beta, int(config["factor_fit_rank"]),
            oversample=int(config["factor_fit_oversample"]),
            power_iterations=int(config["factor_fit_power_iterations"]),
            seed=int(config["seed"]) + 1000 * layer + expert,
            device=device,
        )
        primary_encoded = encode_interaction_factor(
            primary, "int4_per_row", hadamard_rotate=True,
        )
        primary_encoded, primary_scale = make_encoded_factor_self_safe(
            primary_encoded, abc,
        )
        primary_decoded = primary_encoded.decode()

        control = factor_exact_proxy_plus_tail(
            down2, down4, proxy, beta, 0, dtype=np.float32,
        )
        proxy_int8 = encode_interaction_factor(
            control, "int8_per_row", hadamard_rotate=False,
        )
        proxy_int8, _ = make_encoded_factor_self_safe(proxy_int8, abc)
        proxy_int4 = encode_interaction_factor(
            control, "int4_per_row", hadamard_rotate=True,
        )
        proxy_int4, _ = make_encoded_factor_self_safe(proxy_int4, abc)

        rounded_l4 = np.asarray(control.l4, np.float16)
        rounded_l2 = np.asarray(control.l2, np.float16)
        rounded = JointInteractionFactor(
            l4=rounded_l4.astype(np.float32),
            l2=rounded_l2.astype(np.float32),
            method="exact_proxy_fp16",
            tail_rank=0,
            exact_rank=control.rank,
            encoding="fp16",
        )
        _, rounded_scale = make_factor_self_safe(rounded, abc)
        rounded_stored = _stored_downward_scale(rounded_scale)
        rounded_decoded = JointInteractionFactor(
            l4=rounded.l4 * rounded_stored,
            l2=rounded.l2 * rounded_stored,
            method=rounded.method,
            tail_rank=0,
            exact_rank=rounded.rank,
            encoding="fp16_self_safe",
            storage_bytes=8196,
        )
        _, rounded_post = make_factor_self_safe(rounded_decoded, abc)
        if rounded_post != 1.0:
            raise RuntimeError("stored FP16 proxy factor is not self-safe")

        tail = randomized_joint_metric_factor(
            down2, down4, empty_proxy, 0.0,
            int(config["mixed_euclidean_tail_rank"]),
            oversample=int(config["factor_fit_oversample"]),
            power_iterations=int(config["factor_fit_power_iterations"]),
            seed=int(config["seed"]) + 500000 + 1000 * layer + expert,
            device=device,
        )
        tail_encoded = encode_interaction_factor(
            tail, "int4_per_row", hadamard_rotate=True,
        )
        mixed_proxy_encoded = encode_interaction_factor(
            control, "int8_per_row", hadamard_rotate=False,
        )
        tail_decoded = tail_encoded.decode()
        proxy_decoded = mixed_proxy_encoded.decode()
        mixed_decoded = JointInteractionFactor(
            l4=np.concatenate((tail_decoded.l4, proxy_decoded.l4), axis=1),
            l2=np.concatenate((tail_decoded.l2, proxy_decoded.l2), axis=1),
            method="exact_proxy_rank4_int8_plus_euclidean_tail4_int4",
            tail_rank=tail_decoded.rank,
            exact_rank=proxy_decoded.rank,
            encoding="mixed_int8_int4",
            storage_bytes=10244,
        )
        _, mixed_scale_value = make_factor_self_safe(mixed_decoded, abc)
        mixed_stored = _stored_downward_scale(mixed_scale_value)
        mixed_loaded = JointInteractionFactor(
            l4=mixed_decoded.l4 * mixed_stored,
            l2=mixed_decoded.l2 * mixed_stored,
            method=mixed_decoded.method,
            tail_rank=mixed_decoded.tail_rank,
            exact_rank=mixed_decoded.exact_rank,
            encoding=mixed_decoded.encoding + "_self_safe",
            storage_bytes=10244,
        )
        _, mixed_post = make_factor_self_safe(mixed_loaded, abc)
        if mixed_post != 1.0:
            raise RuntimeError("stored mixed factor is not self-safe")

        factor_seconds += time.perf_counter() - started
        expected_payloads = {
            "primary": (primary_encoded.payload_bytes, 6148),
            "rank4_int8": (proxy_int8.payload_bytes, 6148),
            "rank4_int4": (proxy_int4.payload_bytes, 4100),
            "rank4_fp16": (rounded_decoded.payload_bytes, 8196),
            "mixed": (mixed_loaded.payload_bytes, 10244),
        }
        for name, (observed, expected) in expected_payloads.items():
            if int(observed) != int(expected):
                raise RuntimeError(f"{name} factor payload changed")
        if int(control.l4.nbytes + control.l2.nbytes) != 16384:
            raise RuntimeError("rank-4 FP32 control payload changed")
        median, p90 = _fit_probe_error(
            down2, down4, proxy, beta, primary_decoded,
            int(config["seed"]) + 100000 * layer + expert,
        )

        primary_codes.append(np.asarray(primary_encoded.packed_codes, np.uint8))
        primary_scales.append(np.asarray(primary_encoded.row_scales, np.float16))
        primary_global.append(np.float32(primary_encoded.global_scale))
        rank4_l4.append(np.asarray(control.l4, np.float32))
        rank4_l2.append(np.asarray(control.l2, np.float32))
        fp16_l4.append(rounded_l4)
        fp16_l2.append(rounded_l2)
        fp16_global.append(rounded_stored)
        int8_codes.append(np.asarray(proxy_int8.packed_codes, np.uint8))
        int8_scales.append(np.asarray(proxy_int8.row_scales, np.float16))
        int8_global.append(np.float32(proxy_int8.global_scale))
        int4_codes.append(np.asarray(proxy_int4.packed_codes, np.uint8))
        int4_scales.append(np.asarray(proxy_int4.row_scales, np.float16))
        int4_global.append(np.float32(proxy_int4.global_scale))
        mixed_tail_codes.append(np.asarray(tail_encoded.packed_codes, np.uint8))
        mixed_tail_scales.append(np.asarray(tail_encoded.row_scales, np.float16))
        mixed_proxy_codes.append(
            np.asarray(mixed_proxy_encoded.packed_codes, np.uint8)
        )
        mixed_proxy_scales.append(
            np.asarray(mixed_proxy_encoded.row_scales, np.float16)
        )
        mixed_global.append(mixed_stored)
        primary_shrink.append(float(primary_scale))
        fp16_shrink.append(float(rounded_stored))
        mixed_shrink.append(float(mixed_stored))
        probe_median.append(median)
        probe_p90.append(p90)
        del (
            leaves, down2, down4, abc, primary, primary_encoded,
            primary_decoded, control, proxy_int8, proxy_int4, rounded,
            rounded_decoded, tail, tail_encoded, tail_decoded,
            mixed_proxy_encoded, proxy_decoded, mixed_decoded, mixed_loaded,
        )
        if expert % 32 == 31:
            gc.collect()
    arrays = {
        "primary_packed_codes": np.stack(primary_codes),
        "primary_row_scales": np.stack(primary_scales),
        "primary_global_scales": np.asarray(primary_global, np.float32),
        "rank4_l4": np.stack(rank4_l4),
        "rank4_l2": np.stack(rank4_l2),
        "rank4_fp16_l4": np.stack(fp16_l4),
        "rank4_fp16_l2": np.stack(fp16_l2),
        "rank4_fp16_global_scales": np.asarray(fp16_global, np.float32),
        "rank4_int8_packed_codes": np.stack(int8_codes),
        "rank4_int8_row_scales": np.stack(int8_scales),
        "rank4_int8_global_scales": np.asarray(int8_global, np.float32),
        "rank4_int4_packed_codes": np.stack(int4_codes),
        "rank4_int4_row_scales": np.stack(int4_scales),
        "rank4_int4_global_scales": np.asarray(int4_global, np.float32),
        "mixed_tail_int4_packed_codes": np.stack(mixed_tail_codes),
        "mixed_tail_int4_row_scales": np.stack(mixed_tail_scales),
        "mixed_proxy_int8_packed_codes": np.stack(mixed_proxy_codes),
        "mixed_proxy_int8_row_scales": np.stack(mixed_proxy_scales),
        "mixed_global_scales": np.asarray(mixed_global, np.float32),
    }
    diagnostics = {
        "layer": int(layer),
        "experts": int(config["experts_per_layer"]),
        "decode_seconds": decode_seconds,
        "factor_seconds": factor_seconds,
        "primary_self_safe_shrink_min": float(np.min(primary_shrink)),
        "primary_self_safe_shrink_median": float(np.median(primary_shrink)),
        "fp16_self_safe_shrink_min": float(np.min(fp16_shrink)),
        "mixed_self_safe_shrink_min": float(np.min(mixed_shrink)),
        "probe_relative_error_median": float(np.median(probe_median)),
        "probe_relative_error_p90_across_experts": float(
            np.quantile(probe_p90, .9)
        ),
    }
    return arrays, diagnostics

def _fit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _validate_contract(config)
    base_hashes = _verify_base(config, args.pr12_config, args.pr12_dir)
    audit, input_hashes = _locked_inputs(
        config, args.pr10_config, args.exact_captures,
        args.checkpoint, args.trees,
    )
    threads = _thread_contract(config)
    dependency = _dependency_hashes()
    runtime = prior.runtime_provenance(args.device)
    if args.output.exists():
        facts_path = args.output / FIT_FACTS
        if not facts_path.is_file():
            raise RuntimeError("fit output exists without resumable facts")
        facts = json.loads(facts_path.read_text())
        locked = {
            **base_hashes, **dependency,
            "config_sha256": _sha256(args.config),
            "capture_sha256": input_hashes["capture_sha256"],
            "tree_sha256": input_hashes["tree_sha256"],
            "runtime_provenance": runtime,
        }
        for name, value in locked.items():
            if facts.get(name) != value:
                raise RuntimeError(f"fit resume provenance changed: {name}")
        if facts.get("failures") or facts.get("failure_history"):
            raise RuntimeError("failed fit cannot be silently resumed")
        if facts.get("completed") is True:
            return
    else:
        args.output.mkdir(parents=True)
        facts = {
            "completed": False,
            "completed_layers": [],
            "failures": [],
            "failure_history": [],
            "run_id": config["run_id"],
            "phase": "fit",
            "fit_split": "train",
            "validation_rows_admitted_or_used": False,
            "test_scientific_rows_admitted_or_used": False,
            "checkpoint_audit": audit,
            "capture_sha256": input_hashes["capture_sha256"],
            "tree_sha256": input_hashes["tree_sha256"],
            "checkpoint_config_sha256": input_hashes["config_sha256"],
            "checkpoint_index_sha256": input_hashes["index_sha256"],
            "config_sha256": _sha256(args.config),
            "runtime_provenance": runtime,
            "cpu_capacity": _cpu_capacity(),
            "worker_thread_environment": threads,
            **base_hashes,
            **dependency,
        }
        _atomic_json(args.output / FIT_FACTS, facts)
    data = prior.load_capture_admitted(args.exact_captures, ["train"])
    if set(map(str, np.unique(data["split"]))) != {"train"}:
        raise RuntimeError("fit admitted a non-training capture row")
    index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    down_tree = tree_from_record(tree_records["down"])
    layer_records: dict[str, Any] = {}
    existing_manifest = args.output / FIT_MANIFEST
    if existing_manifest.is_file():
        layer_records.update(json.loads(existing_manifest.read_text()).get("layers", {}))
    for completed_layer in map(int, facts.get("completed_layers", [])):
        shard = args.output / FIT_LAYER.format(layer=completed_layer)
        sidecar = args.output / FIT_SIDECAR.format(layer=completed_layer)
        if not shard.is_file() or not sidecar.is_file():
            raise RuntimeError("completed fit layer is missing its shard or sidecar")
        sidecar_payload = json.loads(sidecar.read_text())
        if (
            int(sidecar_payload.get("layer", -1)) != completed_layer
            or sidecar_payload.get("file") != shard.name
            or int(sidecar_payload.get("bytes", -1)) != shard.stat().st_size
            or sidecar_payload.get("sha256") != _sha256(shard)
            or sidecar_payload.get("runtime_provenance") != runtime
        ):
            raise RuntimeError("completed fit layer sidecar provenance changed")
        layer_records[str(completed_layer)] = {
            "file": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": sidecar_payload["sha256"],
            "sidecar": sidecar.name,
            "sidecar_sha256": _sha256(sidecar),
        }
    try:
        for layer in map(int, config["layers"]):
            if layer in set(map(int, facts.get("completed_layers", []))):
                continue
            local = prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)],
                local["split"],
            )
            beta = float(proxy_facts["beta"])
            arrays, diagnostics = _fit_layer(
                config, args.checkpoint, index, down_tree, layer, proxy, beta, args.device,
            )
            arrays["proxy"] = np.asarray(proxy, np.float32)
            arrays["beta"] = np.asarray([beta], np.float64)
            shard = args.output / FIT_LAYER.format(layer=layer)
            _atomic_npz(shard, arrays)
            sidecar = args.output / FIT_SIDECAR.format(layer=layer)
            sidecar_payload = {
                **diagnostics,
                "file": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": _sha256(shard),
                "proxy_facts": proxy_facts,
                "runtime_provenance": runtime,
                "fit_split": "train",
                "validation_rows_admitted_or_used": False,
                "test_scientific_rows_admitted_or_used": False,
                "factor_ids": [
                    str(record["factor_id"]) for record in config["factor_configs"]
                ],
            }
            _atomic_json(sidecar, sidecar_payload)
            layer_records[str(layer)] = {
                "file": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": sidecar_payload["sha256"],
                "sidecar": sidecar.name,
                "sidecar_sha256": _sha256(sidecar),
            }
            facts["completed_layers"] = sorted(
                set(map(int, facts.get("completed_layers", []))) | {layer}
            )
            _atomic_json(args.output / FIT_FACTS, facts)
        if facts["completed_layers"] != list(map(int, config["layers"])):
            raise RuntimeError("factor fit layer grid is incomplete")
        manifest = {
            "run_id": config["run_id"],
            "completed": True,
            "fit_split": "train",
            "experts_per_layer": int(config["experts_per_layer"]),
            "factor_fit_scope": config["factor_fit_scope"],
            "factor_ids": [
                str(record["factor_id"]) for record in config["factor_configs"]
            ],
            "factor_payload_bytes": {
                str(record["factor_id"]): int(record["factor_payload_bytes"])
                for record in config["factor_configs"]
            },
            "primary_factor_payload_bytes": int(config["primary_factor_payload_bytes"]),
            "compute_control_factor_payload_bytes": int(config["compute_control_factor_payload_bytes"]),
            "layers": layer_records,
            "runtime_provenance": runtime,
            "config_sha256": _sha256(args.config),
            "capture_sha256": input_hashes["capture_sha256"],
            "tree_sha256": input_hashes["tree_sha256"],
            "checkpoint_config_sha256": input_hashes["config_sha256"],
            "checkpoint_index_sha256": input_hashes["index_sha256"],
            **base_hashes,
            **dependency,
        }
        _atomic_json(existing_manifest, manifest)
        facts["completed"] = True
        facts["factor_manifest_sha256"] = _sha256(existing_manifest)
        _atomic_json(args.output / FIT_FACTS, facts)
    except Exception as error:
        record = {"type": type(error).__name__, "message": str(error)}
        facts["failures"] = [record]
        facts.setdefault("failure_history", []).append(record)
        _atomic_json(args.output / FIT_FACTS, facts)
        raise


def _load_layer_factors(
    arrays: Mapping[str, np.ndarray], config: Mapping[str, Any], expert: int,
) -> dict[str, JointInteractionFactor]:
    def encoded(
        prefix: str, rank: int, tail_rank: int, exact_rank: int,
        encoding: str, global_name: str | None,
    ) -> EncodedInteractionFactor:
        global_scale = (
            np.float32(1.0) if global_name is None
            else np.float32(arrays[global_name][expert])
        )
        return EncodedInteractionFactor(
            packed_codes=np.asarray(
                arrays[f"{prefix}_packed_codes"][expert], np.uint8,
            ),
            row_scales=np.asarray(
                arrays[f"{prefix}_row_scales"][expert], np.float16,
            ),
            units=UNITS,
            rank=int(rank),
            method=prefix,
            tail_rank=int(tail_rank),
            exact_rank=int(exact_rank),
            encoding=encoding,
            global_scale=global_scale,
        )

    primary_encoded = encoded(
        "primary", int(config["factor_fit_rank"]),
        int(config["factor_fit_rank"]), 0, "int4_hadamard",
        "primary_global_scales",
    )
    rank4_fp32 = JointInteractionFactor(
        l4=np.asarray(arrays["rank4_l4"][expert], np.float32),
        l2=np.asarray(arrays["rank4_l2"][expert], np.float32),
        method="exact_proxy",
        tail_rank=0,
        exact_rank=4,
        encoding="float32",
        storage_bytes=16384,
    )
    fp16_scale = np.float32(arrays["rank4_fp16_global_scales"][expert])
    rank4_fp16 = JointInteractionFactor(
        l4=np.asarray(arrays["rank4_fp16_l4"][expert], np.float16).astype(
            np.float32
        ) * fp16_scale,
        l2=np.asarray(arrays["rank4_fp16_l2"][expert], np.float16).astype(
            np.float32
        ) * fp16_scale,
        method="exact_proxy_fp16",
        tail_rank=0,
        exact_rank=4,
        encoding="fp16_self_safe",
        storage_bytes=8196,
    )
    rank4_int8 = encoded(
        "rank4_int8", 4, 0, 4, "int8_per_row",
        "rank4_int8_global_scales",
    ).decode()
    rank4_int4 = encoded(
        "rank4_int4", 4, 0, 4, "int4_per_row_hadamard",
        "rank4_int4_global_scales",
    ).decode()
    tail = encoded(
        "mixed_tail_int4", 4, 4, 0, "int4_per_row_hadamard", None,
    ).decode()
    proxy_part = encoded(
        "mixed_proxy_int8", 4, 0, 4, "int8_per_row", None,
    ).decode()
    mixed_scale = np.float32(arrays["mixed_global_scales"][expert])
    mixed = JointInteractionFactor(
        l4=np.concatenate((tail.l4, proxy_part.l4), axis=1) * mixed_scale,
        l2=np.concatenate((tail.l2, proxy_part.l2), axis=1) * mixed_scale,
        method="exact_proxy_rank4_int8_plus_euclidean_tail4_int4",
        tail_rank=4,
        exact_rank=4,
        encoding="mixed_int8_int4_self_safe",
        storage_bytes=10244,
    )
    factors = {
        str(config["primary_factor_id"]): primary_encoded.decode(),
        str(config["compute_control_factor_id"]): rank4_fp32,
        "exact_proxy_rank4_fp16": rank4_fp16,
        "exact_proxy_rank4_int8_per_row": rank4_int8,
        "exact_proxy_rank4_int4_per_row_hadamard": rank4_int4,
        "exact_proxy_rank4_int8_plus_euclidean_tail4_int4": mixed,
    }
    expected = {
        str(record["factor_id"]): (
            int(record["rank"]), int(record["factor_payload_bytes"])
        )
        for record in config["factor_configs"]
    }
    if set(factors) != set(expected):
        raise RuntimeError("loaded factor IDs changed")
    for factor_id, factor in factors.items():
        if (factor.rank, factor.payload_bytes) != expected[factor_id]:
            raise RuntimeError(f"loaded factor contract changed: {factor_id}")
    return factors

def _plan(data: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = []
    per_layer: dict[int, int] = {}
    for layer in map(int, config["layers"]):
        local = prior.layer_view(data, layer)
        records = np.flatnonzero(np.asarray(local["split"]).astype(str) == "validation")
        per_layer[layer] = len(records)
        for record in records.tolist():
            experts = np.asarray(local["expert_ids"][record], np.int64)
            weights = np.asarray(local["router_weights"][record], np.float64)
            if experts.shape != (8,) or len(set(experts.tolist())) != 8:
                raise RuntimeError("validation group is not eight unique routed experts")
            if weights.shape != (8,) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=5e-7):
                raise RuntimeError("validation router weights changed")
            groups.append({
                "capture_source": "exact_checkpoint",
                "evaluation_split": "validation",
                "request_id": str(local["request_id"][record]),
                "sequence_id": str(local["sequence_id"][record]),
                "position": int(local["position"][record]),
                "layer": layer,
                "record": int(record),
                "experts": experts.tolist(),
                "router_weights": weights.tolist(),
            })
    if per_layer != {layer: int(config["expected_validation_groups_per_layer"]) for layer in map(int, config["layers"])}:
        raise RuntimeError(f"validation groups per layer changed: {per_layer}")
    if len(groups) != int(config["expected_validation_groups"]):
        raise RuntimeError("validation group count changed")
    identities = {
        tuple(group[name] for name in GROUP_IDENTITY) for group in groups
    }
    if len(identities) != len(groups):
        raise RuntimeError("duplicate validation group identity")
    return groups


def _incremental_solver_macs(
    rank: int, coordinate_sweeps: int, local_passes: int,
    config: Mapping[str, Any],
) -> int:
    coordinate = (
        2 * UNITS * int(rank) + 16 * UNITS
    ) * int(coordinate_sweeps)
    maximum_shortlist = 7 * int(config["local_shortlist"])
    local = int(local_passes) * (
        14 * UNITS * int(rank)
        + maximum_shortlist * maximum_shortlist * int(rank)
    )
    return int(coordinate + local)


def _frontier_compute_macs(
    rank: int, trace: Any, config: Mapping[str, Any],
) -> int:
    build = 16 * UNITS * int(rank) + 24 * UNITS
    return int(build + _incremental_solver_macs(
        rank, int(trace.coordinate_sweeps), int(trace.local_passes), config,
    ))


def _exact_option_data(
    responses: Any, options: Sequence[RateOption], proxy: np.ndarray, beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.stack([
        np.asarray(responses.target_output, np.float64)
        - split_state_output(responses, option.states)
        for option in options
    ])
    features = qmetric_features(residuals, proxy, beta)
    damage = np.einsum("ij,ij->i", features, features, optimize=True)
    return features, damage


def _filter_frontier(
    options: Sequence[RateOption], features: np.ndarray, exact_damage: np.ndarray,
    maximum_pages: int,
) -> tuple[tuple[RateOption, ...], np.ndarray, np.ndarray]:
    keep = [index for index, option in enumerate(options) if int(option.pages) <= int(maximum_pages)]
    if not keep or int(options[keep[0]].pages) != 0:
        raise RuntimeError("burst-cap frontier lost its zero-page endpoint")
    return tuple(options[index] for index in keep), features[keep], exact_damage[keep]


def _uniform_allocation(frontiers: Sequence[Sequence[RateOption]], cap: int) -> AllocationTrace:
    selected = []
    for frontier in frontiers:
        candidates = [
            index for index, option in enumerate(frontier)
            if int(option.pages) <= int(cap)
        ]
        if not candidates:
            raise RuntimeError("uniform allocation has no feasible option")
        selected.append(min(
            candidates,
            key=lambda index: (
                float(frontier[index].damage), int(frontier[index].pages), index,
            ),
        ))
    pages = sum(int(frontiers[e][index].pages) for e, index in enumerate(selected))
    objective = sum(float(frontiers[e][index].damage) for e, index in enumerate(selected))
    return AllocationTrace(np.asarray(selected, np.int64), pages, objective)


def _factor_spec(
    config: Mapping[str, Any], factor_id: str,
) -> Mapping[str, Any]:
    records = [
        record for record in config["factor_configs"]
        if str(record["factor_id"]) == str(factor_id)
    ]
    if len(records) != 1:
        raise RuntimeError(f"factor specification is not unique: {factor_id}")
    return records[0]


def _allocation_rows(
    metadata: Mapping[str, Any], factor_id: str,
    frontiers: Sequence[Sequence[RateOption]], features: Sequence[np.ndarray],
    exact_damage: Sequence[np.ndarray], base_features: Sequence[np.ndarray],
    weights: np.ndarray, mean_pages: int, burst_cap: int, policy: str,
    allocation: AllocationTrace, predicted_objective: float,
    frontier_runtime: float, allocation_runtime: float,
    frontier_macs: int, frontier_dp_evaluations: int,
    frontier_coordinate_sweeps: int, frontier_local_passes: int,
    metadata_bytes: int, config: Mapping[str, Any],
    *, refinement_work: Mapping[str, Any] | None = None,
    global_bound: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = np.asarray(allocation.option_indices, np.int64)
    chosen_features = [features[e][int(index)] for e, index in enumerate(selected)]
    combined = sum(
        (weights[e] * chosen_features[e] for e in range(8)),
        np.zeros(chosen_features[0].shape[0], np.float64),
    )
    base_combined = sum(
        (weights[e] * base_features[e] for e in range(8)),
        np.zeros(base_features[0].shape[0], np.float64),
    )
    group_damage = float(combined @ combined)
    group_base = float(base_combined @ base_combined)
    if not np.isfinite(group_base) or group_base <= 0:
        raise RuntimeError("combined top-8 base damage is not positive finite")
    if policy.startswith("pooled_exact_combined_moe") and not np.isclose(
        group_damage, float(allocation.objective), rtol=1e-9, atol=1e-8,
    ):
        raise RuntimeError("exact group allocation objective changed")
    individual_base = np.asarray([float(value @ value) for value in base_features])
    individual_selected = np.asarray([
        float(exact_damage[e][int(index)]) for e, index in enumerate(selected)
    ])
    additive_base = float(np.sum(weights * weights * individual_base))
    additive_damage = float(np.sum(weights * weights * individual_selected))
    total_pages = int(sum(
        int(frontiers[e][int(index)].pages) for e, index in enumerate(selected)
    ))
    if total_pages != int(allocation.pages) or total_pages > 8 * int(mean_pages):
        raise RuntimeError("group allocation page accounting changed")
    allowed_bytes = 8 * (int(metadata_bytes) + int(mean_pages) * PAGE_BYTES)
    actual_bytes = 8 * int(metadata_bytes) + total_pages * PAGE_BYTES
    refinement = {
        "runtime": 0.0, "rounds": 0, "selected_repairs": 0,
        "adaptive_price_solves": 0, "coordinate_sweeps": 0,
        "local_passes": 0,
        **dict(refinement_work or {}),
    }
    factor_rank = int(_factor_spec(config, factor_id)["rank"])
    extra_macs = _incremental_solver_macs(
        factor_rank, int(refinement["coordinate_sweeps"]),
        int(refinement["local_passes"]), config,
    )
    total_frontier_runtime = float(frontier_runtime) + float(refinement["runtime"])
    total_coordinate_sweeps = (
        int(frontier_coordinate_sweeps)
        + int(refinement["coordinate_sweeps"])
    )
    total_local_passes = (
        int(frontier_local_passes) + int(refinement["local_passes"])
    )
    bound_fields = {
        "global_bound_lower_damage": np.nan,
        "global_bound_upper_damage": np.nan,
        "global_bound_absolute_gap": np.nan,
        "global_bound_relative_gap": np.nan,
        "global_bound_iterations": 0,
        "global_bound_certified": False,
    }
    if global_bound is not None:
        if not np.isclose(
            float(global_bound.upper_bound), group_damage, rtol=1e-9, atol=1e-8,
        ):
            raise RuntimeError("global bound upper value changed")
        bound_fields = {
            "global_bound_lower_damage": float(global_bound.lower_bound),
            "global_bound_upper_damage": float(global_bound.upper_bound),
            "global_bound_absolute_gap": float(global_bound.absolute_gap),
            "global_bound_relative_gap": float(global_bound.relative_gap),
            "global_bound_iterations": int(global_bound.iterations),
            "global_bound_certified": bool(global_bound.certified),
        }
    exact_group = policy.startswith("pooled_exact_combined_moe")
    group = {
        **{name: metadata[name] for name in GROUP_IDENTITY},
        "sequence_id": metadata["sequence_id"],
        "factor_config_id": factor_id,
        "allocation_policy": policy,
        "mean_budget_pages_per_expert": int(mean_pages),
        "group_page_budget": 8 * int(mean_pages),
        "burst_cap_pages_per_expert": int(burst_cap),
        "actual_group_pages": total_pages,
        "unused_group_pages": 8 * int(mean_pages) - total_pages,
        "average_actual_correction_pages": total_pages / 8.0,
        "average_actual_correction_bpw": total_pages / (8.0 * 768.0),
        "average_allowed_correction_bpw": int(mean_pages) / 768.0,
        "factor_payload_bytes_per_expert": (
            int(metadata_bytes) - int(config["abc_metadata_bytes_per_expert"])
        ),
        "abc_metadata_bytes_per_expert": int(
            config["abc_metadata_bytes_per_expert"]
        ),
        "combined_metadata_bytes_per_expert": int(metadata_bytes),
        "combined_metadata_bpw": 8.0 * int(metadata_bytes) / EXPERT_WEIGHTS,
        "average_allowed_total_bpw": allowed_bytes / EXPERT_WEIGHTS,
        "average_actual_total_bpw": actual_bytes / EXPERT_WEIGHTS,
        "strict_one_bpw_pass": (
            allowed_bytes
            <= 8 * int(config["strict_all_in_total_bytes_per_expert"])
        ),
        "group_recovery": 1.0 - group_damage / group_base,
        "group_exact_qenergy_damage": group_damage,
        "group_base_qenergy_damage": group_base,
        "router_square_additive_recovery": 1.0 - additive_damage / additive_base,
        "router_square_additive_damage": additive_damage,
        "router_square_additive_base_damage": additive_base,
        "compressed_predicted_objective": float(predicted_objective),
        "allocation_objective": float(allocation.objective),
        "selected_option_indices": json.dumps(
            selected.tolist(), separators=(",", ":"),
        ),
        "selected_expert_pages": json.dumps([
            int(frontiers[e][int(index)].pages)
            for e, index in enumerate(selected)
        ], separators=(",", ":")),
        "router_weights": json.dumps(weights.tolist(), separators=(",", ":")),
        "frontier_runtime_ms": 1000.0 * total_frontier_runtime,
        "allocation_runtime_ms": 1000.0 * float(allocation_runtime),
        "selector_runtime_ms": 1000.0 * (
            total_frontier_runtime + float(allocation_runtime)
        ),
        "selector_compute_macs": int(frontier_macs) + int(extra_macs),
        "selector_compute_macs_per_expert": (
            int(frontier_macs) + int(extra_macs)
        ) / 8.0,
        "selector_dp_state_evaluations": int(frontier_dp_evaluations),
        "selector_coordinate_sweeps": total_coordinate_sweeps,
        "selector_local_passes": total_local_passes,
        "frontier_refinement_rounds": int(refinement["rounds"]),
        "frontier_selected_repairs": int(refinement["selected_repairs"]),
        "frontier_adaptive_price_solves": int(
            refinement["adaptive_price_solves"]
        ),
        "group_coordinate_sweeps": int(allocation.coordinate_sweeps),
        "group_pair_passes": int(allocation.pair_passes),
        "group_exchange_passes": int(allocation.exchange_passes),
        "group_exchange_evaluations": int(allocation.exchange_evaluations),
        **bound_fields,
        "exact_cross_expert_information_used": exact_group,
        "promotable": False,
        "continuation_eligible": policy in {
            "uniform_per_expert",
            "pooled_router_square_compressed",
            "pooled_router_square_column_generated",
        },
        "selection_regime": (
            "exact_h4_grouped_top8_average_rate_geometry_ceiling"
        ),
    }
    expert_rows = []
    expert_ids = metadata["experts"]
    for router_rank, index in enumerate(selected.tolist()):
        option = frontiers[router_rank][index]
        base_damage = individual_base[router_rank]
        exact = individual_selected[router_rank]
        expert_rows.append({
            **{name: metadata[name] for name in GROUP_IDENTITY},
            "sequence_id": metadata["sequence_id"],
            "expert_id": int(expert_ids[router_rank]),
            "router_rank": router_rank + 1,
            "router_weight": float(weights[router_rank]),
            "factor_config_id": factor_id,
            "allocation_policy": policy,
            "mean_budget_pages_per_expert": int(mean_pages),
            "burst_cap_pages_per_expert": int(burst_cap),
            "selected_pages": int(option.pages),
            "selected_state_counts": json.dumps(
                np.bincount(option.states, minlength=8).tolist(),
                separators=(",", ":"),
            ),
            "selected_states": json.dumps(
                option.states.tolist(), separators=(",", ":"),
            ),
            "option_source": option.source,
            "option_page_price": float(option.page_price),
            "compressed_predicted_damage": float(option.damage),
            "exact_qenergy_damage": exact,
            "base_qenergy_damage": base_damage,
            "expert_recovery": 1.0 - exact / base_damage,
        })
    return group, expert_rows

def _evaluate_group(
    task: int,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if _EVAL_CONTEXT is None:
        raise RuntimeError("evaluation context is not initialized")
    group = _EVAL_CONTEXT["groups"][int(task)]
    config = _EVAL_CONTEXT["config"]
    local = _EVAL_CONTEXT["local"]
    activation = np.asarray(local["x"][int(group["record"])], np.float32)
    weights = np.asarray(group["router_weights"], np.float64)
    expert_ids = list(map(int, group["experts"]))
    responses_by_expert = []
    for expert in expert_ids:
        cell = _EVAL_CONTEXT["experts"][expert]
        responses_by_expert.append(
            split_projection_responses(cell["q2"], cell["q4"], activation)
        )

    factor_data: dict[str, Any] = {}
    diagnostics = {"group_index": int(task), "experts": []}
    factor_ids = [
        str(record["factor_id"]) for record in config["factor_configs"]
    ]
    for factor_id in factor_ids:
        frontiers, features, exact_values, base_features, fields = [], [], [], [], []
        runtime_total = 0.0
        macs_total = dp_evaluations_total = 0
        coordinate_sweeps_total = local_passes_total = 0
        for router_rank, expert in enumerate(expert_ids):
            cell = _EVAL_CONTEXT["experts"][expert]
            responses = responses_by_expert[router_rank]
            field_started = time.perf_counter()
            field = build_split_interaction_field(
                cell["factors"][factor_id], responses.hidden, cell["abc"],
            )
            trace = lagrangian_rate_frontier(
                field,
                maximum_pages=int(config["frontier_maximum_pages"]),
                target_pages=config["frontier_target_pages"],
                price_ratios=config["frontier_price_ratios"],
                coordinate_sweeps=int(config["coordinate_sweeps"]),
                local_shortlist=int(config["local_shortlist"]),
                local_swap_units=int(config["local_swap_units"]),
                local_max_passes=int(config["local_max_passes"]),
                price_local_max_passes=int(config["price_local_max_passes"]),
            )
            runtime_total += time.perf_counter() - field_started
            option_features, option_exact = _exact_option_data(
                responses, trace.options,
                _EVAL_CONTEXT["proxy"], _EVAL_CONTEXT["beta"],
            )
            zero = next(
                index for index, option in enumerate(trace.options)
                if int(option.pages) == 0
            )
            fields.append(field)
            frontiers.append(trace.options)
            features.append(option_features)
            exact_values.append(option_exact)
            base_features.append(option_features[zero])
            macs_total += _frontier_compute_macs(
                cell["factors"][factor_id].rank, trace, config,
            )
            dp_evaluations_total += int(trace.diagonal_dp_state_evaluations)
            coordinate_sweeps_total += int(trace.coordinate_sweeps)
            local_passes_total += int(trace.local_passes)
            diagnostics["experts"].append({
                "expert_id": expert,
                "factor_config_id": factor_id,
                "frontier_options": len(trace.options),
                "coordinate_sweeps": int(trace.coordinate_sweeps),
                "local_passes": int(trace.local_passes),
            })
        factor_data[factor_id] = {
            "fields": tuple(fields),
            "frontiers": tuple(frontiers),
            "features": tuple(features),
            "exact": tuple(exact_values),
            "base": tuple(base_features),
            "responses": tuple(responses_by_expert),
            "runtime": runtime_total,
            "macs": macs_total,
            "dp_evaluations": dp_evaluations_total,
            "coordinate_sweeps": coordinate_sweeps_total,
            "local_passes": local_passes_total,
        }

    group_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []

    def emit(
        factor_id: str, mean_pages: int, burst: int, policy: str,
        allocation: AllocationTrace, allocation_seconds: float,
        filtered: tuple[Any, Any, Any],
        *, refinement_trace: Any | None = None,
        refinement_seconds: float = 0.0,
        global_bound: Any | None = None,
    ) -> None:
        frontiers, features, exact_values = filtered
        data = factor_data[factor_id]
        predicted = float(sum(
            weights[e] * weights[e] * frontiers[e][int(index)].damage
            for e, index in enumerate(allocation.option_indices)
        ))
        spec = _factor_spec(config, factor_id)
        metadata_bytes = (
            int(config["abc_metadata_bytes_per_expert"])
            + int(spec["factor_payload_bytes"])
        )
        refinement_work = None
        if refinement_trace is not None:
            refinement_work = {
                "runtime": float(refinement_seconds),
                "rounds": int(refinement_trace.rounds),
                "selected_repairs": int(refinement_trace.selected_repairs),
                "adaptive_price_solves": int(
                    refinement_trace.adaptive_price_solves
                ),
                "coordinate_sweeps": int(refinement_trace.coordinate_sweeps),
                "local_passes": int(refinement_trace.local_passes),
            }
        row, experts = _allocation_rows(
            group, factor_id, frontiers, features, exact_values, data["base"],
            weights, mean_pages, burst, policy, allocation, predicted,
            data["runtime"], allocation_seconds, data["macs"],
            data["dp_evaluations"], data["coordinate_sweeps"],
            data["local_passes"], metadata_bytes, config,
            refinement_work=refinement_work, global_bound=global_bound,
        )
        group_rows.append(row)
        expert_rows.extend(experts)

    def filtered(
        factor_id: str, maximum_pages: int,
    ) -> tuple[Any, Any, Any]:
        data = factor_data[factor_id]
        return tuple(zip(*[
            _filter_frontier(
                data["frontiers"][expert], data["features"][expert],
                data["exact"][expert], maximum_pages,
            )
            for expert in range(8)
        ]))

    def refined(
        factor_id: str, mean_pages: int, burst: int,
    ) -> tuple[tuple[Any, Any, Any], Any, float]:
        data = factor_data[factor_id]
        coarse = filtered(factor_id, burst)
        started = time.perf_counter()
        trace = allocation_aware_rate_frontiers(
            data["fields"], coarse[0],
            page_budget=8 * int(mean_pages),
            burst_cap_pages=int(burst),
            weights=weights * weights,
            coordinate_sweeps=int(config["coordinate_sweeps"]),
            local_shortlist=int(config["local_shortlist"]),
            local_swap_units=int(config["local_swap_units"]),
            local_max_passes=int(config["local_max_passes"]),
            max_rounds=int(config["column_generation_max_rounds"]),
            adaptive_price_multipliers=config[
                "column_generation_price_multipliers"
            ],
            adaptive_price_local_passes=int(
                config["column_generation_price_local_max_passes"]
            ),
        )
        refined_features, refined_exact = [], []
        for expert, options in enumerate(trace.frontiers):
            values, damage = _exact_option_data(
                data["responses"][expert], options,
                _EVAL_CONTEXT["proxy"], _EVAL_CONTEXT["beta"],
            )
            refined_features.append(values)
            refined_exact.append(damage)
        seconds = time.perf_counter() - started
        return (
            (trace.frontiers, tuple(refined_features), tuple(refined_exact)),
            trace,
            seconds,
        )

    primary_id = str(config["primary_factor_id"])
    for mean_pages in map(int, config["mean_correction_page_budgets"]):
        uniform_filtered = filtered(primary_id, mean_pages)
        started = time.perf_counter()
        uniform = _uniform_allocation(uniform_filtered[0], mean_pages)
        emit(
            primary_id, mean_pages, mean_pages, "uniform_per_expert",
            uniform, time.perf_counter() - started, uniform_filtered,
        )

        coarse_by_cap = {
            burst: filtered(primary_id, burst)
            for burst in map(int, config["primary_burst_caps_pages"])
        }
        refined_full = refinement_full = refinement_seconds_full = None
        router_seed_for_exact = None
        for burst in map(int, config["primary_burst_caps_pages"]):
            coarse = coarse_by_cap[burst]
            started = time.perf_counter()
            allocation = multiple_choice_allocate(
                coarse[0], 8 * mean_pages, weights * weights,
            )
            emit(
                primary_id, mean_pages, burst,
                "pooled_router_square_compressed", allocation,
                time.perf_counter() - started, coarse,
            )
            if burst == int(config["global_bound_burst_cap_pages"]):
                router_seed_for_exact = allocation

            refined_data, refinement, refinement_seconds = refined(
                primary_id, mean_pages, burst,
            )
            emit(
                primary_id, mean_pages, burst,
                "pooled_router_square_column_generated",
                refinement.allocation, 0.0, refined_data,
                refinement_trace=refinement,
                refinement_seconds=refinement_seconds,
            )
            if burst == int(config["global_bound_burst_cap_pages"]):
                refined_full = refined_data
                refinement_full = refinement
                refinement_seconds_full = refinement_seconds

        full = coarse_by_cap[int(config["global_bound_burst_cap_pages"])]
        started = time.perf_counter()
        equal = multiple_choice_allocate(
            full[0], 8 * mean_pages, np.ones(8),
        )
        emit(
            primary_id, mean_pages, 1536,
            "pooled_equal_weight_compressed", equal,
            time.perf_counter() - started, full,
        )
        if router_seed_for_exact is None:
            raise RuntimeError("missing router-weight exact-oracle seed")
        started = time.perf_counter()
        exact_coarse = exact_group_option_allocate(
            full[0], full[1], weights, 8 * mean_pages,
            router_seed_for_exact.option_indices,
            max_coordinate_sweeps=int(config["group_exact_coordinate_sweeps"]),
            max_pair_passes=int(config["group_exact_pair_passes"]),
        )
        emit(
            primary_id, mean_pages, 1536,
            "pooled_exact_combined_moe_oracle", exact_coarse,
            time.perf_counter() - started, full,
        )
        if (
            refined_full is None or refinement_full is None
            or refinement_seconds_full is None
        ):
            raise RuntimeError("missing full-burst refined frontier")
        started = time.perf_counter()
        exact_refined = exact_group_option_allocate(
            refined_full[0], refined_full[1], weights, 8 * mean_pages,
            refinement_full.allocation.option_indices,
            max_coordinate_sweeps=int(config["group_exact_coordinate_sweeps"]),
            max_pair_passes=int(config["group_exact_pair_passes"]),
        )
        emit(
            primary_id, mean_pages, 1536,
            "pooled_exact_combined_moe_column_generated_local",
            exact_refined, time.perf_counter() - started, refined_full,
            refinement_trace=refinement_full,
            refinement_seconds=refinement_seconds_full,
        )

        if mean_pages == int(config["global_bound_mean_pages"]):
            started = time.perf_counter()
            exchanged = bounded_group_exchange_allocate(
                refined_full[0], refined_full[1], weights,
                8 * mean_pages, exact_refined,
                exchange_sizes=config["global_exchange_sizes"],
                shortlist_size=int(config["global_exchange_shortlist"]),
                max_passes=int(config["global_exchange_max_passes"]),
            )
            bound = global_group_dual_bound(
                refined_full[0], refined_full[1], weights,
                8 * mean_pages, exchanged,
                max_iterations=int(config["global_bound_max_iterations"]),
                relative_tolerance=float(
                    config["global_bound_relative_tolerance"]
                ),
            )
            emit(
                primary_id, mean_pages, 1536,
                "pooled_exact_combined_moe_global_bound",
                exchanged, time.perf_counter() - started, refined_full,
                refinement_trace=refinement_full,
                refinement_seconds=refinement_seconds_full,
                global_bound=bound,
            )

    for spec in config["factor_configs"]:
        factor_id = str(spec["factor_id"])
        if factor_id == primary_id:
            continue
        mean_pages = int(spec["all_in_mean_pages"])
        full = filtered(factor_id, 1536)
        uniform_filtered = filtered(factor_id, mean_pages)
        started = time.perf_counter()
        uniform = _uniform_allocation(uniform_filtered[0], mean_pages)
        emit(
            factor_id, mean_pages, mean_pages, "uniform_per_expert",
            uniform, time.perf_counter() - started, uniform_filtered,
        )
        started = time.perf_counter()
        router = multiple_choice_allocate(
            full[0], 8 * mean_pages, weights * weights,
        )
        emit(
            factor_id, mean_pages, 1536,
            "pooled_router_square_compressed", router,
            time.perf_counter() - started, full,
        )
        started = time.perf_counter()
        exact_coarse = exact_group_option_allocate(
            full[0], full[1], weights, 8 * mean_pages,
            router.option_indices,
            max_coordinate_sweeps=int(config["group_exact_coordinate_sweeps"]),
            max_pair_passes=int(config["group_exact_pair_passes"]),
        )
        emit(
            factor_id, mean_pages, 1536,
            "pooled_exact_combined_moe_oracle", exact_coarse,
            time.perf_counter() - started, full,
        )
        refined_data, refinement, refinement_seconds = refined(
            factor_id, mean_pages, 1536,
        )
        emit(
            factor_id, mean_pages, 1536,
            "pooled_router_square_column_generated",
            refinement.allocation, 0.0, refined_data,
            refinement_trace=refinement,
            refinement_seconds=refinement_seconds,
        )
        started = time.perf_counter()
        exact_refined = exact_group_option_allocate(
            refined_data[0], refined_data[1], weights, 8 * mean_pages,
            refinement.allocation.option_indices,
            max_coordinate_sweeps=int(config["group_exact_coordinate_sweeps"]),
            max_pair_passes=int(config["group_exact_pair_passes"]),
        )
        emit(
            factor_id, mean_pages, 1536,
            "pooled_exact_combined_moe_column_generated_local",
            exact_refined, time.perf_counter() - started, refined_data,
            refinement_trace=refinement,
            refinement_seconds=refinement_seconds,
        )
    return int(task), group_rows, expert_rows, diagnostics

def _evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _validate_contract(config)
    base_hashes = _verify_base(config, args.pr12_config, args.pr12_dir)
    audit, input_hashes = _locked_inputs(
        config, args.pr10_config, args.exact_captures,
        args.checkpoint, args.trees,
    )
    threads = _thread_contract(config)
    dependency = _dependency_hashes()
    runtime = prior.runtime_provenance(args.device)
    fit_facts_path = args.fit_dir / FIT_FACTS
    manifest_path = args.fit_dir / FIT_MANIFEST
    fit_facts = json.loads(fit_facts_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if fit_facts.get("completed") is not True or manifest.get("completed") is not True:
        raise RuntimeError("factor fit is incomplete")
    for name, value in {
        "config_sha256": _sha256(args.config),
        "capture_sha256": input_hashes["capture_sha256"],
        "tree_sha256": input_hashes["tree_sha256"],
        "checkpoint_config_sha256": input_hashes["config_sha256"],
        "checkpoint_index_sha256": input_hashes["index_sha256"],
        "runtime_provenance": runtime,
        **base_hashes,
        **dependency,
    }.items():
        if fit_facts.get(name) != value or manifest.get(name) != value:
            raise RuntimeError(f"fit/evaluation provenance changed: {name}")
    if fit_facts.get("factor_manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("fit manifest hash changed")
    data = prior.load_capture_admitted(args.exact_captures, ["train", "validation"])
    if "test" in set(map(str, np.unique(data["split"]))):
        raise RuntimeError("test row entered average-rate evaluation")
    groups = _plan(data, config)
    frozen_evaluation = {
        "run_id": config["run_id"],
        "phase": "evaluate",
        "selection_split": "validation",
        "test_scientific_rows_admitted_or_used": False,
        "validation_groups": len(groups),
        "validation_expert_invocations": 8 * len(groups),
        "grouping_policy": config["grouping_policy"],
        "capture_sha256": input_hashes["capture_sha256"],
        "tree_sha256": input_hashes["tree_sha256"],
        "checkpoint_config_sha256": input_hashes["config_sha256"],
        "checkpoint_index_sha256": input_hashes["index_sha256"],
        "config_sha256": _sha256(args.config),
        "fit_facts_sha256": _sha256(fit_facts_path),
        "fit_manifest_sha256": _sha256(manifest_path),
        "runtime_provenance": runtime,
        "evaluation_workers": int(config["evaluation_workers"]),
        "worker_thread_environment": threads,
        **base_hashes,
        **dependency,
    }
    group_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    if args.output.exists():
        facts_path = args.output / RUN_FACTS
        if not facts_path.is_file():
            raise RuntimeError("evaluation output exists without resumable facts")
        facts = json.loads(facts_path.read_text())
        for name, value in frozen_evaluation.items():
            if facts.get(name) != value:
                raise RuntimeError(f"evaluation resume provenance changed: {name}")
        if facts.get("failures") or facts.get("failure_history"):
            raise RuntimeError("failed evaluation cannot be silently resumed")
        if facts.get("completed") is True:
            return
        completed = set(map(int, facts.get("completed_layers", [])))
        group_path = args.output / GROUP_FRONTIER
        expert_path = args.output / EXPERT_ALLOCATION
        if completed:
            if not group_path.is_file() or not expert_path.is_file():
                raise RuntimeError("completed evaluation layer lacks checkpoint Parquets")
            group_frame = pd.read_parquet(group_path)
            expert_frame = pd.read_parquet(expert_path)
            if set(map(int, group_frame["layer"].unique())) != completed:
                raise RuntimeError("group checkpoint layer scope changed")
            if set(map(int, expert_frame["layer"].unique())) != completed:
                raise RuntimeError("expert checkpoint layer scope changed")
            expected_groups = (
                len(completed)
                * int(config["expected_validation_groups_per_layer"])
                * len(_expected_policy_grid(config))
            )
            if len(group_frame) != expected_groups or len(expert_frame) != 8 * expected_groups:
                raise RuntimeError("evaluation checkpoint row count changed")
            group_rows = group_frame.to_dict("records")
            expert_rows = expert_frame.to_dict("records")
            diagnostics = list(facts.get("frontier_diagnostics", []))
            if len(diagnostics) != len(completed) * int(
                config["expected_validation_groups_per_layer"]
            ):
                raise RuntimeError("evaluation checkpoint diagnostics changed")
        elif (group_path.exists() or expert_path.exists()):
            raise RuntimeError("uncommitted evaluation Parquet is present")
    else:
        args.output.mkdir(parents=True)
        facts = {
            "completed": False,
            "completed_layers": [],
            "failures": [],
            "failure_history": [],
            "checkpoint_audit": audit,
            "cpu_capacity": _cpu_capacity(),
            "frontier_diagnostics": [],
            **frozen_evaluation,
        }
        _atomic_json(args.output / RUN_FACTS, facts)
    index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in PROJECTIONS}
    try:
        for layer in map(int, config["layers"]):
            if layer in set(map(int, facts.get("completed_layers", []))):
                continue
            local = prior.layer_view(data, layer)
            proxy, proxy_facts = base.proxy_gradients(
                local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)],
                local["split"],
            )
            beta = float(proxy_facts["beta"])
            record = manifest["layers"][str(layer)]
            shard = args.fit_dir / record["file"]
            sidecar = args.fit_dir / record["sidecar"]
            if (
                _sha256(shard) != record["sha256"]
                or shard.stat().st_size != int(record["bytes"])
                or _sha256(sidecar) != record["sidecar_sha256"]
            ):
                raise RuntimeError("factor shard/sidecar changed")
            arrays = dict(np.load(shard, allow_pickle=False))
            if not np.array_equal(np.asarray(arrays["proxy"], np.float32), np.asarray(proxy, np.float32)):
                raise RuntimeError("train-derived proxy changed")
            if not np.isclose(float(arrays["beta"][0]), beta, rtol=0, atol=0):
                raise RuntimeError("train-derived proxy beta changed")
            layer_groups = [group for group in groups if int(group["layer"]) == layer]
            active = sorted({int(expert) for group in layer_groups for expert in group["experts"]})
            experts = {}
            for expert in active:
                decoded = prior.decode_expert(args.checkpoint, index, trees, layer, expert)
                q2 = tuple(decoded[name][0] for name in PROJECTIONS)
                q4 = tuple(decoded[name][2] for name in PROJECTIONS)
                experts[expert] = {
                    "q2": q2,
                    "q4": q4,
                    "abc": unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta),
                    "factors": _load_layer_factors(arrays, config, expert),
                }
            global _EVAL_CONTEXT
            _EVAL_CONTEXT = {
                "groups": tuple(layer_groups), "local": local,
                "experts": experts, "proxy": proxy, "beta": beta,
                "config": config,
            }
            pending: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
            context = mp.get_context(config["worker_start_method"])
            workers = min(int(config["evaluation_workers"]), len(layer_groups))
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                futures = {
                    executor.submit(_evaluate_group, index): index
                    for index in range(len(layer_groups))
                }
                for future in as_completed(futures):
                    index_value, rows_value, expert_value, diagnostics_value = future.result()
                    if index_value != futures[future]:
                        raise RuntimeError("worker group identity changed")
                    pending[index_value] = (rows_value, expert_value, diagnostics_value)
            for index_value in range(len(layer_groups)):
                rows_value, expert_value, diagnostics_value = pending[index_value]
                group_rows.extend(rows_value)
                expert_rows.extend(expert_value)
                diagnostics.append(diagnostics_value)
            _atomic_parquet(args.output / GROUP_FRONTIER, group_rows)
            _atomic_parquet(args.output / EXPERT_ALLOCATION, expert_rows)
            facts["completed_layers"] = sorted(
                set(map(int, facts["completed_layers"])) | {layer}
            )
            facts["observed_group_rows"] = len(group_rows)
            facts["observed_expert_rows"] = len(expert_rows)
            facts["frontier_diagnostics"] = diagnostics
            _atomic_json(args.output / RUN_FACTS, facts)
            del experts, arrays
            _EVAL_CONTEXT = None
            gc.collect()
        expected_group_rows = (
            int(config["expected_validation_groups"])
            * len(_expected_policy_grid(config))
        )
        expected_expert_rows = expected_group_rows * 8
        if len(group_rows) != expected_group_rows or len(expert_rows) != expected_expert_rows:
            raise RuntimeError(
                f"row grid changed: group={len(group_rows)}/{expected_group_rows}, "
                f"expert={len(expert_rows)}/{expected_expert_rows}"
            )
        group_frame = pd.DataFrame(group_rows)
        expert_frame = pd.DataFrame(expert_rows)
        if group_frame[list(GROUP_IDENTITY) + [
            "factor_config_id", "allocation_policy", "mean_budget_pages_per_expert",
            "burst_cap_pages_per_expert",
        ]].duplicated().any():
            raise RuntimeError("duplicate group frontier row")
        if np.any(group_frame["actual_group_pages"] > group_frame["group_page_budget"]):
            raise RuntimeError("group page budget exceeded")
        if np.any(expert_frame["selected_pages"] > expert_frame["burst_cap_pages_per_expert"]):
            raise RuntimeError("per-expert burst cap exceeded")
        accounting = {
            "schema_version": 2,
            "group_rows": len(group_rows),
            "expert_rows": len(expert_rows),
            "validation_groups": int(config["expected_validation_groups"]),
            "experts_per_group": 8,
            "page_bytes": PAGE_BYTES,
            "expert_weights": EXPERT_WEIGHTS,
            "primary_combined_metadata_bytes": int(config["primary_factor_payload_bytes"])
            + int(config["abc_metadata_bytes_per_expert"]),
            "compute_control_combined_metadata_bytes": int(config["compute_control_factor_payload_bytes"])
            + int(config["abc_metadata_bytes_per_expert"]),
            "factor_payload_bytes": {
                str(record["factor_id"]): int(record["factor_payload_bytes"])
                for record in config["factor_configs"]
            },
            "factor_all_in_mean_pages": {
                str(record["factor_id"]): int(record["all_in_mean_pages"])
                for record in config["factor_configs"]
            },
            "policies_per_group": len(_expected_policy_grid(config)),
            "mean_correction_page_budgets": config["mean_correction_page_budgets"],
            "primary_burst_caps_pages": config["primary_burst_caps_pages"],
            "frontier_diagnostics": diagnostics,
        }
        _atomic_json(args.output / ACCOUNTING, accounting)
        facts["completed"] = True
        facts["group_frontier_sha256"] = _sha256(args.output / GROUP_FRONTIER)
        facts["expert_allocation_sha256"] = _sha256(args.output / EXPERT_ALLOCATION)
        facts["accounting_sha256"] = _sha256(args.output / ACCOUNTING)
        _atomic_json(args.output / RUN_FACTS, facts)
    except Exception as error:
        record = {"type": type(error).__name__, "message": str(error)}
        facts["failures"] = [record]
        facts.setdefault("failure_history", []).append(record)
        _atomic_json(args.output / RUN_FACTS, facts)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr12-config", type=Path, required=True)
    parser.add_argument("--pr10-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--pr12-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "fit":
        _fit(args, config)
    else:
        if args.fit_dir is None:
            parser.error("--fit-dir is required for evaluate")
        _evaluate(args, config)


if __name__ == "__main__":
    main()
