from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_average_rate_all_layers",
    ROOT / "scripts/run_average_rate_all_layers.py",
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def _config() -> dict:
    return json.loads(
        (ROOT / "configs/qwen36_mxfp4_average_rate_all_layers.json").read_text()
    )


def _base_config() -> dict:
    return json.loads(
        (ROOT / "configs/qwen36_mxfp4_average_rate_allocation.json").read_text()
    )


def test_all_layer_contract_inherits_the_complete_pr13_grid():
    config = _config()
    runner._validate_contract(
        config,
        _base_config(),
        ROOT / "configs/qwen36_mxfp4_average_rate_allocation.json",
    )
    assert config["layers"] == list(range(40))
    assert len(runner.pr13._expected_policy_grid(config)) == 66
    assert config["expected_validation_groups"] == 1280
    changed = dict(config)
    changed["coordinate_sweeps"] = 7
    with pytest.raises(RuntimeError, match="scientific contract"):
        runner._validate_contract(
            changed,
            _base_config(),
            ROOT / "configs/qwen36_mxfp4_average_rate_allocation.json",
        )


def test_layer_plan_uses_every_complete_validation_top8_group():
    config = _config()
    rows = 102
    local = {
        "split": np.asarray(["train"] * 70 + ["validation"] * 32),
        "request_id": np.asarray([f"r{index}" for index in range(rows)]),
        "sequence_id": np.asarray([f"s{index}" for index in range(rows)]),
        "position": np.arange(rows),
        "expert_ids": np.tile(np.arange(8), (rows, 1)),
        "router_weights": np.tile(np.ones(8) / 8, (rows, 1)),
    }
    groups = runner._layer_plan(local, config, 17)
    assert len(groups) == 32
    assert {group["layer"] for group in groups} == {17}
    assert all(len(group["experts"]) == 8 for group in groups)
    local["expert_ids"][70, 1] = 0
    with pytest.raises(RuntimeError, match="eight unique"):
        runner._layer_plan(local, config, 17)


def test_test_capture_shard_is_rejected_before_loading(monkeypatch, tmp_path: Path):
    opened = []
    monkeypatch.setattr(runner.np, "load", lambda *args, **kwargs: opened.append(args))
    manifest = {"splits": {"test": {"0": {"file": "test_layer_00.npz"}}}}
    with pytest.raises(RuntimeError, match="sealed"):
        runner._load_capture_layer(tmp_path, manifest, "test", 0)
    assert opened == []


def test_combine_capture_splits_never_admits_test_rows():
    train = {
        "split": np.asarray(["train", "train"]),
        "layer": np.asarray([0, 0]),
        "request_id": np.asarray(["train-0", "train-1"]),
    }
    validation = {
        "split": np.asarray(["validation"]),
        "layer": np.asarray([0]),
        "request_id": np.asarray(["validation-0"]),
    }
    combined = runner._combine_capture_splits(train, validation)
    assert combined["split"].tolist() == ["train", "train", "validation"]
    assert "test" not in combined["split"]


def test_sharded_admission_rejects_cross_split_request_leakage():
    data = {
        "split": np.asarray(["train", "validation"]),
        "request_id": np.asarray(["same-request", "same-request"]),
    }
    with pytest.raises(RuntimeError, match="request split leakage"):
        runner._admit_sharded_capture_rows(data, ["train", "validation"])


def test_evaluation_layer_paths_are_disjoint():
    root = Path("/tmp/result")
    zero = runner._eval_paths(root, 0)
    one = runner._eval_paths(root, 1)
    assert not set(zero) & set(one)
    assert all(path.parent == root / "layers" for path in zero)
