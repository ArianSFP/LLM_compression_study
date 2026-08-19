"""Micro tests for follow-up row accounting and analyzer materialization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_sparse_streaming_tile_followup as followup
from oracle_study.sparse_streaming_analysis import validate_run_facts
from oracle_study.sparse_streaming_cuda import SelectionTrace


def _config() -> dict:
    return {
        "physical_budgets_bpw": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        "reference_bpw": 4.25,
        "suffix_bpw_per_complete_representation": 2.0,
        "page_size_bytes": 512,
        "reference": {
            "revision": "locked-revision",
            "config_sha256": "checkpoint-config",
            "index_sha256": "checkpoint-index",
        },
        "locked_tree_sha256": "tree",
        "locked_capture_sha256": {"exact_checkpoint": "capture"},
    }


def _metadata() -> dict:
    return {
        "capture_source": "exact_checkpoint",
        "evaluation_split": "validation",
        "request_id": "validation-request",
        "sequence_id": "validation-sequence",
        "position": 7,
        "layer": 4,
        "expert_id": 154,
        "expert_stratum": "hot",
        "router_rank": 1,
        "router_coefficient": 0.5,
    }


def _mock_followup_rows(monkeypatch) -> list[dict]:
    budgets = [followup.base.physical_page_budget(value) for value in _config()["physical_budgets_bpw"]]

    def trace(*_args, **_kwargs) -> SelectionTrace:
        maximum = max(budgets)
        return SelectionTrace(
            order=torch.arange(maximum, dtype=torch.int64),
            gains=torch.ones(maximum, dtype=torch.float64),
            snapshots={page: torch.zeros(2) for page in budgets},
        )

    monkeypatch.setattr(followup, "selector_trace", trace)
    monkeypatch.setattr(followup, "static_proxy_mixed_page_order", trace)
    monkeypatch.setattr(followup.base, "assert_complete_endpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        followup.base, "decoded_complete_outputs",
        lambda *_args, **_kwargs: (np.zeros(2), np.ones(2)),
    )
    monkeypatch.setattr(followup.base, "recovery", lambda *_args, **_kwargs: 0.75)
    matrices = {
        projection: [np.zeros((1, 1), np.float32) for _ in range(3)]
        for projection in followup.base.PROJECTIONS
    }
    return followup.followup_rows(
        matrices, np.ones(1, np.float32), np.ones((2, 1), np.float32),
        1.0, _metadata(), _config(), torch.device("cpu"),
    )


def test_followup_rows_have_complete_selector_grid_and_physical_accounting(monkeypatch) -> None:
    rows = _mock_followup_rows(monkeypatch)
    frame = pd.DataFrame(rows)
    assert len(frame) == len(followup.FOLLOWUP_SPECS) * 6
    assert set(zip(frame.tile_shape, frame.selector)) == {
        spec.identity for spec in followup.FOLLOWUP_SPECS
    }
    assert set(frame.evaluation_split) == {"validation"}
    assert np.array_equal(frame.physical_bytes, frame.physical_pages * 512)
    assert np.allclose(frame.logical_bpw, frame.physical_budget_bpw)
    assert np.allclose(frame.page_amplification, 1.0)
    exact = frame[frame.conditional_followup_selector == "exact_dynamic_tile_marginal"]
    static = frame[frame.conditional_followup_selector != "exact_dynamic_tile_marginal"]
    assert np.array_equal(exact.global_refreshes, exact.physical_pages)
    assert set(static.global_refreshes) == {1}
    assert not (set(followup.base.ARTIFACT_SCHEMAS["tile_streaming_frontier"]) - set(frame))


def test_flush_materializes_analyzer_compatible_validation_only_directory(
    tmp_path: Path, monkeypatch,
) -> None:
    rows = _mock_followup_rows(monkeypatch)
    parent = tmp_path / "parent"
    output = tmp_path / "followup"
    parent.mkdir()
    output.mkdir()
    for name, filename in followup.base.ARTIFACT_FILES.items():
        followup.base.atomic_parquet(
            parent / filename, [], followup.base.ARTIFACT_SCHEMAS[name],
        )
    followup.base.atomic_parquet(
        parent / "_support_cache.parquet", [],
        [*followup.base.IDENTITY_COLUMNS, "sequence_id", "projection", "score_method",
         "shortlist_size", "support_action_count", "support_budget_bpw", "action_ids",
         "action_scores"],
    )
    facts = {
        "run_id": "synthetic-followup",
        "run_mode": "pilot",
        "config_sha256": "config",
        "reference": _config()["reference"],
        "locked_tree_sha256": "tree",
        "locked_capture_sha256": {"exact_checkpoint": "capture"},
        "page_size_bytes": 512,
        "codec_locked": True,
        "request_separation_verified": True,
        "expected_unique_invocations": 1,
    }
    followup.flush(output, rows, facts, final=True, schema_parent=parent)
    written = json.loads((output / "run_facts.json").read_text())
    assert written["completed"] is True
    assert written["observed_unique_invocations"] == 1
    assert written["artifact_row_counts"]["tile_streaming_frontier"] == len(rows)
    validate_run_facts(written, _config(), "config")
    for name, filename in followup.base.ARTIFACT_FILES.items():
        frame = pd.read_parquet(output / filename)
        if name == "tile_streaming_frontier":
            assert len(frame) == len(rows)
        else:
            assert frame.empty
            assert set(followup.base.ARTIFACT_SCHEMAS[name]).issubset(frame.columns)
