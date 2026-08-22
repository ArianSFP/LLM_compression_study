#!/usr/bin/env python3
"""Distributed all-layer extension of the frozen PR #13 average-rate study.

The scientific kernels are imported unchanged from run_average_rate_allocation.
This driver adds hash-locked sharded capture loading, one-layer atomic work units,
and deterministic finalizers suitable for multiple heterogeneous workers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.mxfp4_embed import audit_checkpoint  # noqa: E402
from oracle_study.neuron_selector import unit_score_metadata  # noqa: E402
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_average_rate_allocation as pr13  # noqa: E402
import run_set_utility_distillation as set_study  # noqa: E402
import run_sparse_streaming_study as sparse  # noqa: E402


SCHEMA = "average_rate_all_layers_distributed_v1"
CAPTURE_SCHEMA = "exact_mxfp4_all_layer_sharded_v1"
LAYERS = tuple(range(40))
SPLIT_ROWS = {"train": 70, "validation": 32, "test": 39}
CAPTURE_ARRAYS = (
    "sequence_id", "request_id", "position", "layer", "block_type", "split",
    "prefix_hash", "x", "xplus", "router_logits", "expert_ids",
    "router_weights", "h1_xplus", "h1_router_logits", "h1_expert_ids",
    "h2_xplus", "h2_router_logits", "h2_expert_ids", "h3_xplus",
    "h3_router_logits", "h3_expert_ids", "h4_xplus", "h4_router_logits",
    "h4_expert_ids",
)

FIT_FAILURE = "average_rate_factor_layer_{layer}.failure.json"
EVAL_GROUP_LAYER = "average_rate_group_frontier_layer_{layer}.parquet"
EVAL_EXPERT_LAYER = "average_rate_expert_allocation_layer_{layer}.parquet"
EVAL_SIDECAR = "average_rate_evaluation_layer_{layer}.json"
EVAL_FAILURE = "average_rate_evaluation_layer_{layer}.failure.json"

INHERITED_CHANGEABLE = {
    "run_id", "layers", "expected_validation_groups",
    "expected_validation_expert_invocations",
}
POLICY_IDENTITY = (
    "factor_config_id", "allocation_policy", "mean_budget_pages_per_expert",
    "burst_cap_pages_per_expert",
)


def sha256(path: Path) -> str:
    return set_study.sha256(Path(path))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_study.atomic_json(path, payload)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pr13._atomic_npz(path, arrays)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _dependency_hashes() -> dict[str, str]:
    paths = {
        "runner_sha256": Path(__file__).resolve(),
        "base_pr13_runner_sha256": EXPERIMENT / "scripts/run_average_rate_allocation.py",
        "allocator_core_sha256": EXPERIMENT / "src/oracle_study/average_rate_allocator.py",
        "split_core_sha256": EXPERIMENT / "src/oracle_study/split_interaction_field.py",
        "interaction_core_sha256": EXPERIMENT / "src/oracle_study/interaction_field.py",
        "selector_core_sha256": EXPERIMENT / "src/oracle_study/neuron_selector.py",
        "mxfp4_core_sha256": EXPERIMENT / "src/oracle_study/mxfp4_embed.py",
        "set_utility_runner_sha256": EXPERIMENT / "scripts/run_set_utility_distillation.py",
        "sparse_runner_sha256": EXPERIMENT / "scripts/run_sparse_streaming_study.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def _validate_contract(
    config: Mapping[str, Any], base_config: Mapping[str, Any],
    base_config_path: Path,
) -> None:
    pr13._validate_contract(base_config)
    if sha256(base_config_path) != str(config.get("base_pr13_config_sha256")):
        raise RuntimeError("frozen PR #13 config changed")
    dependencies = _dependency_hashes()
    if dependencies["base_pr13_runner_sha256"] != str(
        config.get("base_pr13_runner_sha256")
    ):
        raise RuntimeError("frozen PR #13 runner changed")
    if config.get("base_pr13_commit") != "dfba3748e51d6916dc1f28cb1d3b3250188adbbe":
        raise RuntimeError("all-layer study is not stacked on frozen PR #13")
    if list(map(int, config.get("layers", []))) != list(LAYERS):
        raise RuntimeError("all-layer grid changed")
    if int(config.get("expected_validation_groups_per_layer", -1)) != 32:
        raise RuntimeError("validation groups per layer changed")
    if int(config.get("expected_validation_groups", -1)) != 1280:
        raise RuntimeError("all-layer validation group count changed")
    if int(config.get("expected_validation_expert_invocations", -1)) != 10240:
        raise RuntimeError("all-layer expert invocation count changed")
    if config.get("capture_schema") != CAPTURE_SCHEMA:
        raise RuntimeError("all-layer capture schema changed")
    if not _valid_sha256(config.get("capture_manifest_sha256")):
        raise RuntimeError("capture manifest hash is invalid")
    if not _valid_sha256(config.get("capture_identity_grid_sha256")):
        raise RuntimeError("capture identity-grid hash is invalid")
    if not _valid_sha256(config.get("selected_tree_sha256")):
        raise RuntimeError("selected-tree hash is invalid")
    for key, expected in base_config.items():
        if key in INHERITED_CHANGEABLE:
            continue
        if config.get(key) != expected:
            raise RuntimeError(f"PR #13 scientific contract changed: {key}")
    if len(pr13._expected_policy_grid(config)) != 66:
        raise RuntimeError("PR #13 policy grid changed")


def _validate_capture_manifest(
    manifest: Mapping[str, Any], config: Mapping[str, Any], manifest_path: Path,
) -> None:
    if sha256(manifest_path) != str(config["capture_manifest_sha256"]):
        raise RuntimeError("all-layer capture manifest changed")
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise RuntimeError("capture schema changed")
    if list(map(int, manifest.get("layers", []))) != list(LAYERS):
        raise RuntimeError("capture layer grid changed")
    if manifest.get("array_names") != list(CAPTURE_ARRAYS):
        raise RuntimeError("capture array schema changed")
    if manifest.get("identity_grid_sha256") != config["capture_identity_grid_sha256"]:
        raise RuntimeError("capture identity grid changed")
    expected = manifest.get("expected_grid", {})
    if expected != {
        "layers": 40,
        "rows_per_layer": 141,
        "shards": 120,
        "split_rows_per_layer": {"test": 39, "train": 70, "validation": 32},
        "total_rows": 5640,
    }:
        raise RuntimeError("capture row/shard grid changed")
    checkpoint = manifest.get("checkpoint", {})
    if (
        checkpoint.get("config_sha256") != config["checkpoint_config_sha256"]
        or checkpoint.get("index_sha256") != config["checkpoint_index_sha256"]
        or checkpoint.get("revision") != config["checkpoint_revision"]
        or int(checkpoint.get("num_hidden_layers", -1)) != 40
    ):
        raise RuntimeError("capture checkpoint provenance changed")
    boundary = manifest.get("test_boundary")
    if boundary != {
        "test_shards_written": True,
        "test_rows_admitted_to_evaluator": False,
        "test_scientific_values_used": False,
        "evaluator_allowed_splits": ["train", "validation"],
    }:
        raise RuntimeError("capture test boundary changed")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_ROWS):
        raise RuntimeError("capture split grid changed")
    seen: set[str] = set()
    for split, rows in SPLIT_ROWS.items():
        records = splits[split]
        if not isinstance(records, Mapping) or set(records) != set(map(str, LAYERS)):
            raise RuntimeError(f"capture {split} layer grid changed")
        for layer in LAYERS:
            record = records[str(layer)]
            name = str(record.get("file"))
            if name != f"{split}_layer_{layer:02d}.npz" or Path(name).name != name:
                raise RuntimeError("capture shard filename changed")
            if name in seen or not _valid_sha256(record.get("sha256")):
                raise RuntimeError("capture shard identity changed")
            seen.add(name)
            if int(record.get("rows", -1)) != rows or int(record.get("bytes", -1)) <= 0:
                raise RuntimeError("capture shard row/byte count changed")
            if record.get("array_names") != list(CAPTURE_ARRAYS):
                raise RuntimeError("capture shard array schema changed")


def _capture_record(
    manifest: Mapping[str, Any], split: str, layer: int,
) -> Mapping[str, Any]:
    if split == "test":
        raise RuntimeError("test capture shards are sealed from this study")
    return manifest["splits"][split][str(int(layer))]


def _load_capture_layer(
    capture_dir: Path, manifest: Mapping[str, Any], split: str, layer: int,
) -> dict[str, np.ndarray]:
    record = _capture_record(manifest, split, layer)
    name = Path(str(record["file"]))
    if name.is_absolute() or name.name != str(name):
        raise RuntimeError("capture shard path must be a local basename")
    path = capture_dir / name
    if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
        raise RuntimeError(f"capture {split} layer {layer} shard changed")
    with np.load(path, allow_pickle=False) as loaded:
        if tuple(loaded.files) != CAPTURE_ARRAYS:
            raise RuntimeError("capture shard member order/schema changed")
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows = len(arrays["split"])
    if rows != int(record["rows"]):
        raise RuntimeError("capture shard row count changed")
    if set(map(str, np.unique(arrays["split"]))) != {split}:
        raise RuntimeError("capture shard split label changed")
    if set(map(int, np.unique(arrays["layer"]))) != {int(layer)}:
        raise RuntimeError("capture shard layer label changed")
    return _admit_sharded_capture_rows(arrays, [split])


def _admit_sharded_capture_rows(
    data: Mapping[str, np.ndarray], allowed_splits: Iterable[str],
) -> dict[str, np.ndarray]:
    """Validate and copy already-split shards without weakening the test seal.

    PR #13 monolithic-capture helper requires all three split labels to be
    physically present before it filters out test. This runner deliberately
    does not stage test shards, so it verifies the admitted train/validation
    request partition directly after the locked manifest and shard hashes have
    been checked.
    """
    allowed = frozenset(map(str, allowed_splits))
    if not allowed or not allowed <= {"train", "validation"} or "test" in allowed:
        raise RuntimeError("all-layer runner cannot admit test rows")
    split = np.asarray(data["split"]).astype(str)
    request = np.asarray(data["request_id"]).astype(str)
    if split.shape != request.shape:
        raise RuntimeError("request split metadata arrays must have equal shape")
    observed = set(map(str, np.unique(split)))
    if observed != set(allowed):
        raise RuntimeError(f"capture split labels changed: {observed} != {set(allowed)}")
    request_splits: dict[str, str] = {}
    for request_id, label in zip(request, split):
        previous = request_splits.setdefault(str(request_id), str(label))
        if previous != str(label):
            raise RuntimeError(f"request split leakage for {request_id}")
    rows = len(split)
    result: dict[str, np.ndarray] = {}
    for name, raw in data.items():
        value = np.asarray(raw)
        result[name] = value.copy()
        if value.ndim and value.shape[0] != rows:
            raise RuntimeError(f"capture array {name} row count changed")
    return result


def _combine_capture_splits(
    train: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if tuple(train) != tuple(validation):
        raise RuntimeError("train/validation capture schemas differ")
    combined = {
        name: np.concatenate((np.asarray(train[name]), np.asarray(validation[name])), axis=0)
        for name in train
    }
    return _admit_sharded_capture_rows(combined, ["train", "validation"])


def _common_runtime(device: str) -> dict[str, Any]:
    runtime = dict(set_study.runtime_provenance(device))
    runtime.pop("host", None)
    runtime["schema"] = SCHEMA
    return runtime


def _driver_version() -> str | None:
    try:
        value = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()[0].strip()
        return value or None
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _execution_provenance(device: str, worker_id: str) -> dict[str, Any]:
    return {
        **set_study.runtime_provenance(device),
        "worker_id": str(worker_id),
        "platform": platform.platform(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "nvidia_driver": _driver_version(),
        "cpu_capacity": pr13._cpu_capacity(),
    }


def _thread_contract(workers: int) -> dict[str, str | None]:
    values = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if set(values.values()) != {"1"}:
        raise RuntimeError("BLAS thread environment must be frozen to one")
    capacity = pr13._cpu_capacity()
    if capacity["quota_cores"] is not None and capacity["quota_cores"] < workers:
        raise RuntimeError("CPU quota is below requested evaluation workers")
    return values


def _load_inputs(args: argparse.Namespace) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    config = _load_json(args.config)
    base_config = _load_json(args.base_pr13_config)
    _validate_contract(config, base_config, args.base_pr13_config)
    manifest_path = args.capture_dir / "capture_manifest.json"
    manifest = _load_json(manifest_path)
    _validate_capture_manifest(manifest, config, manifest_path)
    if sha256(args.trees) != str(config["selected_tree_sha256"]):
        raise RuntimeError("selected trees changed")
    checkpoint_hashes = {
        "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
        "checkpoint_index_sha256": sha256(
            args.checkpoint / "model.safetensors.index.json"
        ),
    }
    if checkpoint_hashes != {
        "checkpoint_config_sha256": config["checkpoint_config_sha256"],
        "checkpoint_index_sha256": config["checkpoint_index_sha256"],
    }:
        raise RuntimeError("checkpoint config/index changed")
    dependencies = _dependency_hashes()
    shared = {
        "config_sha256": sha256(args.config),
        "capture_sha256": sha256(manifest_path),
        "tree_sha256": sha256(args.trees),
        **checkpoint_hashes,
        "runtime_provenance": _common_runtime(args.device),
        "base_pr12_config_sha256": config["base_pr12_config_sha256"],
        "base_pr12_runner_sha256": config["base_pr12_runner_sha256"],
        "base_pr12_core_sha256": config["base_pr12_core_sha256"],
        "base_pr12_run_facts_sha256": config["base_pr12_run_facts_sha256"],
        "base_pr12_frontier_sha256": config["base_pr12_frontier_sha256"],
        "base_pr13_commit": config["base_pr13_commit"],
        "base_pr13_config_sha256": config["base_pr13_config_sha256"],
        **dependencies,
    }
    return config, base_config, manifest, shared


def _require_shared(payload: Mapping[str, Any], shared: Mapping[str, Any], label: str) -> None:
    for name, expected in shared.items():
        if payload.get(name) != expected:
            raise RuntimeError(f"{label} provenance changed: {name}")


def verify_inputs(args: argparse.Namespace) -> None:
    config, _, manifest, shared = _load_inputs(args)
    audit = audit_checkpoint(args.checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(f"checkpoint audit failed: {audit}")
    admitted_bytes = 0
    for split in ("train", "validation"):
        for layer in LAYERS:
            record = _capture_record(manifest, split, layer)
            path = args.capture_dir / str(record["file"])
            if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
                raise RuntimeError(f"capture {split} layer {layer} shard changed")
            admitted_bytes += path.stat().st_size
    staged_test_shards = [
        str(manifest["splits"]["test"][str(layer)]["file"])
        for layer in LAYERS
        if (args.capture_dir / str(
            manifest["splits"]["test"][str(layer)]["file"]
        )).exists()
    ]
    if staged_test_shards:
        raise RuntimeError("sealed test shards must not be staged on evaluator workers")
    print(json.dumps({
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "checkpoint_audit": audit,
        "admitted_capture_bytes_verified": admitted_bytes,
        "sealed_test_shards_staged_or_opened": 0,
        "shared_provenance": shared,
    }, indent=2, sort_keys=True))


def _fit_paths(fit_dir: Path, layer: int) -> tuple[Path, Path, Path]:
    return (
        fit_dir / pr13.FIT_LAYER.format(layer=layer),
        fit_dir / pr13.FIT_SIDECAR.format(layer=layer),
        fit_dir / FIT_FAILURE.format(layer=layer),
    )


def fit_layer(args: argparse.Namespace) -> None:
    config, _, manifest, shared = _load_inputs(args)
    layer = int(args.layer)
    if layer not in LAYERS:
        raise RuntimeError("fit layer is outside range(40)")
    shard, sidecar, failure = _fit_paths(args.fit_dir, layer)
    if failure.exists():
        raise RuntimeError(f"prior fit failure must be reviewed: {failure}")
    if shard.exists() or sidecar.exists():
        if not shard.is_file() or not sidecar.is_file():
            raise RuntimeError("partial fit artifact exists")
        payload = _load_json(sidecar)
        _require_shared(payload, shared, "fit-layer resume")
        if (
            payload.get("completed") is not True
            or int(payload.get("layer", -1)) != layer
            or payload.get("sha256") != sha256(shard)
            or int(payload.get("bytes", -1)) != shard.stat().st_size
        ):
            raise RuntimeError("fit-layer resume artifact changed")
        return
    args.fit_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        train = _load_capture_layer(args.capture_dir, manifest, "train", layer)
        proxy, proxy_facts = sparse.proxy_gradients(
            train["xplus"], [train[f"h{h}_router_logits"] for h in range(1, 5)],
            train["split"],
        )
        beta = float(proxy_facts["beta"])
        index = _load_json(args.checkpoint / "model.safetensors.index.json")["weight_map"]
        tree_records = _load_json(args.trees)
        down_tree = tree_from_record(tree_records["down"])
        arrays, diagnostics = pr13._fit_layer(
            config, args.checkpoint, index, down_tree, layer, proxy, beta, args.device,
        )
        arrays["proxy"] = np.asarray(proxy, np.float32)
        arrays["beta"] = np.asarray([beta], np.float64)
        _atomic_npz(shard, arrays)
        payload = {
            "completed": True,
            "schema": SCHEMA,
            "run_id": config["run_id"],
            "phase": "fit-layer",
            "layer": layer,
            "file": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": sha256(shard),
            "fit_split": "train",
            "validation_rows_admitted_or_used": False,
            "test_scientific_rows_admitted_or_used": False,
            "capture_train_shard_sha256": manifest["splits"]["train"][str(layer)]["sha256"],
            "proxy_facts": proxy_facts,
            "factor_ids": [str(record["factor_id"]) for record in config["factor_configs"]],
            "wall_seconds": time.perf_counter() - started,
            "execution_provenance": _execution_provenance(args.device, args.worker_id),
            **diagnostics,
            **shared,
        }
        _atomic_json(sidecar, payload)
    except Exception as error:
        _atomic_json(failure, {
            "completed": False, "schema": SCHEMA, "phase": "fit-layer",
            "layer": layer, "worker_id": args.worker_id,
            "error": {"type": type(error).__name__, "message": str(error)},
            **shared,
        })
        raise


def finalize_fit(args: argparse.Namespace) -> None:
    config, _, _, shared = _load_inputs(args)
    audit = audit_checkpoint(args.checkpoint, config["layers"])
    if not audit["passed"]:
        raise RuntimeError(f"checkpoint audit failed: {audit}")
    layer_records: dict[str, Any] = {}
    execution: dict[str, Any] = {}
    for layer in LAYERS:
        shard, sidecar, failure = _fit_paths(args.fit_dir, layer)
        if failure.exists():
            raise RuntimeError(f"fit failure is present: {failure}")
        if not shard.is_file() or not sidecar.is_file():
            raise RuntimeError(f"fit layer {layer} is incomplete")
        payload = _load_json(sidecar)
        _require_shared(payload, shared, f"fit layer {layer}")
        if (
            payload.get("completed") is not True
            or int(payload.get("layer", -1)) != layer
            or payload.get("sha256") != sha256(shard)
            or int(payload.get("bytes", -1)) != shard.stat().st_size
        ):
            raise RuntimeError(f"fit layer {layer} artifact changed")
        layer_records[str(layer)] = {
            "file": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": sha256(shard),
            "sidecar": sidecar.name,
            "sidecar_sha256": sha256(sidecar),
        }
        execution[str(layer)] = payload["execution_provenance"]
    manifest = {
        "completed": True,
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "fit_split": "train",
        "experts_per_layer": int(config["experts_per_layer"]),
        "factor_fit_scope": config["factor_fit_scope"],
        "factor_ids": [str(record["factor_id"]) for record in config["factor_configs"]],
        "factor_payload_bytes": {
            str(record["factor_id"]): int(record["factor_payload_bytes"])
            for record in config["factor_configs"]
        },
        "primary_factor_payload_bytes": int(config["primary_factor_payload_bytes"]),
        "compute_control_factor_payload_bytes": int(config["compute_control_factor_payload_bytes"]),
        "layers": layer_records,
        "execution_provenance_by_layer": execution,
        **shared,
    }
    manifest_path = args.fit_dir / pr13.FIT_MANIFEST
    _atomic_json(manifest_path, manifest)
    facts = {
        "completed": True,
        "completed_layers": list(LAYERS),
        "failures": [],
        "failure_history": [],
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "phase": "fit",
        "fit_split": "train",
        "validation_rows_admitted_or_used": False,
        "test_scientific_rows_admitted_or_used": False,
        "checkpoint_audit": audit,
        "factor_manifest_sha256": sha256(manifest_path),
        "execution_provenance_by_layer": execution,
        **shared,
    }
    _atomic_json(args.fit_dir / pr13.FIT_FACTS, facts)


def _layer_plan(
    local: Mapping[str, np.ndarray], config: Mapping[str, Any], layer: int,
) -> list[dict[str, Any]]:
    records = np.flatnonzero(np.asarray(local["split"]).astype(str) == "validation")
    if len(records) != int(config["expected_validation_groups_per_layer"]):
        raise RuntimeError(f"validation groups changed for layer {layer}")
    groups = []
    for record in records.tolist():
        experts = np.asarray(local["expert_ids"][record], np.int64)
        weights = np.asarray(local["router_weights"][record], np.float64)
        if experts.shape != (8,) or len(set(experts.tolist())) != 8:
            raise RuntimeError("validation group is not eight unique routed experts")
        if (
            weights.shape != (8,) or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1.0, atol=5e-7)
        ):
            raise RuntimeError("validation router weights changed")
        groups.append({
            "capture_source": "exact_checkpoint",
            "evaluation_split": "validation",
            "request_id": str(local["request_id"][record]),
            "sequence_id": str(local["sequence_id"][record]),
            "position": int(local["position"][record]),
            "layer": int(layer),
            "record": int(record),
            "experts": experts.tolist(),
            "router_weights": weights.tolist(),
        })
    identities = {tuple(group[name] for name in pr13.GROUP_IDENTITY) for group in groups}
    if len(identities) != len(groups):
        raise RuntimeError("duplicate validation group identity")
    return groups


def _eval_paths(validation_dir: Path, layer: int) -> tuple[Path, Path, Path, Path]:
    layer_dir = validation_dir / "layers"
    return (
        layer_dir / EVAL_GROUP_LAYER.format(layer=layer),
        layer_dir / EVAL_EXPERT_LAYER.format(layer=layer),
        layer_dir / EVAL_SIDECAR.format(layer=layer),
        layer_dir / EVAL_FAILURE.format(layer=layer),
    )


def evaluate_layer(args: argparse.Namespace) -> None:
    config, _, manifest, shared = _load_inputs(args)
    layer = int(args.layer)
    workers = int(args.workers)
    if layer not in LAYERS:
        raise RuntimeError("evaluation layer is outside range(40)")
    if workers < 1 or workers > 24:
        raise RuntimeError("evaluation workers must be in [1,24]")
    threads = _thread_contract(workers)
    group_path, expert_path, sidecar_path, failure = _eval_paths(
        args.validation_dir, layer
    )
    if failure.exists():
        raise RuntimeError(f"prior evaluation failure must be reviewed: {failure}")
    if group_path.exists() or expert_path.exists() or sidecar_path.exists():
        if not group_path.is_file() or not expert_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError("partial evaluation artifact exists")
        payload = _load_json(sidecar_path)
        _require_shared(payload, shared, "evaluation-layer resume")
        if (
            payload.get("completed") is not True
            or int(payload.get("layer", -1)) != layer
            or payload.get("group_sha256") != sha256(group_path)
            or payload.get("expert_sha256") != sha256(expert_path)
        ):
            raise RuntimeError("evaluation-layer resume artifact changed")
        return
    shard, fit_sidecar_path, fit_failure = _fit_paths(args.fit_dir, layer)
    if fit_failure.exists() or not shard.is_file() or not fit_sidecar_path.is_file():
        raise RuntimeError(f"fit layer {layer} is incomplete")
    fit_sidecar = _load_json(fit_sidecar_path)
    _require_shared(fit_sidecar, shared, f"fit layer {layer}")
    if fit_sidecar.get("sha256") != sha256(shard):
        raise RuntimeError("factor shard changed before evaluation")
    group_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        train = _load_capture_layer(args.capture_dir, manifest, "train", layer)
        validation = _load_capture_layer(
            args.capture_dir, manifest, "validation", layer
        )
        local = _combine_capture_splits(train, validation)
        proxy, proxy_facts = sparse.proxy_gradients(
            local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)],
            local["split"],
        )
        beta = float(proxy_facts["beta"])
        arrays = dict(np.load(shard, allow_pickle=False))
        if not np.array_equal(np.asarray(arrays["proxy"], np.float32), np.asarray(proxy, np.float32)):
            raise RuntimeError("train-derived proxy changed")
        if not np.isclose(float(arrays["beta"][0]), beta, rtol=0, atol=0):
            raise RuntimeError("train-derived proxy beta changed")
        groups = _layer_plan(local, config, layer)
        index = _load_json(args.checkpoint / "model.safetensors.index.json")["weight_map"]
        tree_records = _load_json(args.trees)
        trees = {
            name: tree_from_record(tree_records[name]) for name in pr13.PROJECTIONS
        }
        active = sorted({int(expert) for group in groups for expert in group["experts"]})
        experts = {}
        for expert in active:
            decoded = set_study.decode_expert(
                args.checkpoint, index, trees, layer, expert
            )
            q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
            q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
            experts[expert] = {
                "q2": q2,
                "q4": q4,
                "abc": unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta),
                "factors": pr13._load_layer_factors(arrays, config, expert),
            }
        pr13._EVAL_CONTEXT = {
            "groups": tuple(groups), "local": local, "experts": experts,
            "proxy": proxy, "beta": beta, "config": config,
        }
        pending: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
        context = mp.get_context(config["worker_start_method"])
        effective_workers = min(workers, len(groups))
        with ProcessPoolExecutor(
            max_workers=effective_workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(pr13._evaluate_group, index): index
                for index in range(len(groups))
            }
            for future in as_completed(futures):
                index_value, rows_value, expert_value, diagnostics_value = future.result()
                if index_value != futures[future]:
                    raise RuntimeError("worker group identity changed")
                pending[index_value] = (rows_value, expert_value, diagnostics_value)
        group_rows, expert_rows, diagnostics = [], [], []
        for index_value in range(len(groups)):
            rows_value, expert_value, diagnostics_value = pending[index_value]
            group_rows.extend(rows_value)
            expert_rows.extend(expert_value)
            diagnostics.append(diagnostics_value)
        expected_groups = 32 * len(pr13._expected_policy_grid(config))
        if len(group_rows) != expected_groups or len(expert_rows) != 8 * expected_groups:
            raise RuntimeError("evaluation layer row grid changed")
        group_frame = pd.DataFrame(group_rows)
        expert_frame = pd.DataFrame(expert_rows)
        _atomic_parquet(group_path, group_frame)
        _atomic_parquet(expert_path, expert_frame)
        payload = {
            "completed": True,
            "schema": SCHEMA,
            "run_id": config["run_id"],
            "phase": "evaluate-layer",
            "layer": layer,
            "selection_split": "validation",
            "test_scientific_rows_admitted_or_used": False,
            "validation_groups": 32,
            "group_rows": len(group_frame),
            "expert_rows": len(expert_frame),
            "group_file": group_path.name,
            "group_bytes": group_path.stat().st_size,
            "group_sha256": sha256(group_path),
            "expert_file": expert_path.name,
            "expert_bytes": expert_path.stat().st_size,
            "expert_sha256": sha256(expert_path),
            "factor_sha256": sha256(shard),
            "factor_sidecar_sha256": sha256(fit_sidecar_path),
            "capture_validation_shard_sha256": manifest["splits"]["validation"][str(layer)]["sha256"],
            "effective_workers": effective_workers,
            "worker_thread_environment": threads,
            "active_experts": len(active),
            "frontier_diagnostics": diagnostics,
            "wall_seconds": time.perf_counter() - started,
            "execution_provenance": _execution_provenance(args.device, args.worker_id),
            **shared,
        }
        _atomic_json(sidecar_path, payload)
        del experts, arrays
        pr13._EVAL_CONTEXT = None
        gc.collect()
    except Exception as error:
        pr13._EVAL_CONTEXT = None
        _atomic_json(failure, {
            "completed": False, "schema": SCHEMA, "phase": "evaluate-layer",
            "layer": layer, "worker_id": args.worker_id,
            "error": {"type": type(error).__name__, "message": str(error)},
            **shared,
        })
        raise


def finalize_evaluate(args: argparse.Namespace) -> None:
    config, _, _, shared = _load_inputs(args)
    fit_facts_path = args.fit_dir / pr13.FIT_FACTS
    fit_manifest_path = args.fit_dir / pr13.FIT_MANIFEST
    fit_facts = _load_json(fit_facts_path)
    fit_manifest = _load_json(fit_manifest_path)
    _require_shared(fit_facts, shared, "fit facts")
    _require_shared(fit_manifest, shared, "fit manifest")
    if fit_facts.get("completed") is not True or fit_manifest.get("completed") is not True:
        raise RuntimeError("factor fit is incomplete")
    group_frames, expert_frames, diagnostics = [], [], []
    execution: dict[str, Any] = {}
    worker_counts: dict[str, int] = {}
    for layer in LAYERS:
        group_path, expert_path, sidecar_path, failure = _eval_paths(
            args.validation_dir, layer
        )
        if failure.exists():
            raise RuntimeError(f"evaluation failure is present: {failure}")
        if not group_path.is_file() or not expert_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError(f"evaluation layer {layer} is incomplete")
        payload = _load_json(sidecar_path)
        _require_shared(payload, shared, f"evaluation layer {layer}")
        if (
            payload.get("completed") is not True
            or int(payload.get("layer", -1)) != layer
            or payload.get("group_sha256") != sha256(group_path)
            or payload.get("expert_sha256") != sha256(expert_path)
        ):
            raise RuntimeError(f"evaluation layer {layer} artifact changed")
        group_frames.append(pd.read_parquet(group_path))
        expert_frames.append(pd.read_parquet(expert_path))
        diagnostics.extend(payload["frontier_diagnostics"])
        execution[str(layer)] = payload["execution_provenance"]
        worker_counts[str(layer)] = int(payload["effective_workers"])
    groups = pd.concat(group_frames, ignore_index=True)
    experts = pd.concat(expert_frames, ignore_index=True)
    group_sort = list(pr13.GROUP_IDENTITY) + list(POLICY_IDENTITY)
    expert_sort = group_sort + ["router_rank"]
    groups = groups.sort_values(group_sort, kind="stable").reset_index(drop=True)
    experts = experts.sort_values(expert_sort, kind="stable").reset_index(drop=True)
    expected_group_rows = int(config["expected_validation_groups"]) * len(
        pr13._expected_policy_grid(config)
    )
    expected_expert_rows = 8 * expected_group_rows
    if len(groups) != expected_group_rows or len(experts) != expected_expert_rows:
        raise RuntimeError("aggregate row grid changed")
    if groups[group_sort].duplicated().any() or experts[expert_sort].duplicated().any():
        raise RuntimeError("aggregate output contains duplicate rows")
    if np.any(groups["actual_group_pages"] > groups["group_page_budget"]):
        raise RuntimeError("group page budget exceeded")
    if np.any(experts["selected_pages"] > experts["burst_cap_pages_per_expert"]):
        raise RuntimeError("expert burst cap exceeded")
    args.validation_dir.mkdir(parents=True, exist_ok=True)
    group_output = args.validation_dir / pr13.GROUP_FRONTIER
    expert_output = args.validation_dir / pr13.EXPERT_ALLOCATION
    _atomic_parquet(group_output, groups)
    _atomic_parquet(expert_output, experts)
    accounting = {
        "schema_version": 2,
        "distributed_schema": SCHEMA,
        "group_rows": len(groups),
        "expert_rows": len(experts),
        "validation_groups": int(config["expected_validation_groups"]),
        "experts_per_group": 8,
        "page_bytes": pr13.PAGE_BYTES,
        "expert_weights": pr13.EXPERT_WEIGHTS,
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
        "policies_per_group": len(pr13._expected_policy_grid(config)),
        "mean_correction_page_budgets": config["mean_correction_page_budgets"],
        "primary_burst_caps_pages": config["primary_burst_caps_pages"],
        "frontier_diagnostics": diagnostics,
        "evaluation_workers_by_layer": worker_counts,
    }
    accounting_path = args.validation_dir / pr13.ACCOUNTING
    _atomic_json(accounting_path, accounting)
    facts = {
        "completed": True,
        "completed_layers": list(LAYERS),
        "failures": [],
        "failure_history": [],
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "phase": "evaluate",
        "selection_split": "validation",
        "test_scientific_rows_admitted_or_used": False,
        "validation_groups": int(config["expected_validation_groups"]),
        "validation_expert_invocations": int(config["expected_validation_expert_invocations"]),
        "grouping_policy": config["grouping_policy"],
        "fit_facts_sha256": sha256(fit_facts_path),
        "fit_manifest_sha256": sha256(fit_manifest_path),
        "group_frontier_sha256": sha256(group_output),
        "expert_allocation_sha256": sha256(expert_output),
        "accounting_sha256": sha256(accounting_path),
        "observed_group_rows": len(groups),
        "observed_expert_rows": len(experts),
        "frontier_diagnostics": diagnostics,
        "evaluation_workers": "distributed",
        "evaluation_workers_by_layer": worker_counts,
        "execution_provenance_by_layer": execution,
        **shared,
    }
    _atomic_json(args.validation_dir / pr13.RUN_FACTS, facts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=(
            "verify-inputs", "fit-layer", "finalize-fit",
            "evaluate-layer", "finalize-evaluate",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-pr13-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--worker-id", default=platform.node())
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.phase in {"fit-layer", "evaluate-layer"} and args.layer is None:
        parser.error(f"--layer is required for {args.phase}")
    if args.phase in {"fit-layer", "finalize-fit", "evaluate-layer", "finalize-evaluate"} and args.fit_dir is None:
        parser.error(f"--fit-dir is required for {args.phase}")
    if args.phase in {"evaluate-layer", "finalize-evaluate"} and args.validation_dir is None:
        parser.error(f"--validation-dir is required for {args.phase}")
    actions = {
        "verify-inputs": verify_inputs,
        "fit-layer": fit_layer,
        "finalize-fit": finalize_fit,
        "evaluate-layer": evaluate_layer,
        "finalize-evaluate": finalize_evaluate,
    }
    actions[args.phase](args)


if __name__ == "__main__":
    main()
