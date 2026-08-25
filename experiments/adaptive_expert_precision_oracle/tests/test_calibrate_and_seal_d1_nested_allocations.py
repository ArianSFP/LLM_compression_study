from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

import calibrate_and_seal_d1_nested_allocations as cli  # noqa: E402
from oracle_study.d1_nested_artifacts import (  # noqa: E402
    ArtifactValidationError,
    canonical_json_bytes,
    freeze_calibration_spec,
)


LAYERS = (0, 1, 4, 6, 12, 23)
WINDOWS = [16, 32, 64]
ETAS = [0.0, 0.0001]
RATES = [360, 384, 725, 749]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any, *, canonical: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        canonical_json_bytes(value)
        if canonical
        else (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _string_set_sha256(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def _protocol_files(tmp_path: Path) -> dict[str, Any]:
    config = {
        "schema": cli.ALLOCATION_CONFIG_SCHEMA,
        "injection_layers": list(LAYERS),
        "request_source": {
            "calibration_requests": 1,
            "evaluation_requests": 1,
        },
        "common_core": {"repair_window_pages_per_group_grid": WINDOWS},
        "local_guardrail": {"eta_grid": ETAS},
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
    }
    config_path = tmp_path / "allocation.json"
    config_sha = _write_json(config_path, config)

    rows = [
        {
            "schema": cli.REQUEST_MANIFEST_SCHEMA,
            "request_id": "request-calibration",
            "prompt_sha256": hashlib.sha256(b"calibration").hexdigest(),
            "domain": "reasoning",
            "decode_position": 8,
            "split": "calibration",
        },
        {
            "schema": cli.REQUEST_MANIFEST_SCHEMA,
            "request_id": "request-evaluation",
            "prompt_sha256": hashlib.sha256(b"evaluation").hexdigest(),
            "domain": "reasoning",
            "decode_position": 8,
            "split": "evaluation",
        },
    ]
    manifest_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    manifest_path = tmp_path / "requests.jsonl"
    manifest_path.write_bytes(manifest_payload)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    calibration_ids = ["request-calibration"]
    evaluation_ids = ["request-evaluation"]
    calibration_prompts = [rows[0]["prompt_sha256"]]
    evaluation_prompts = [rows[1]["prompt_sha256"]]
    facts = {
        "schema": cli.REQUEST_FACTS_SCHEMA,
        "completed": True,
        "output": {
            "sha256": manifest_sha,
            "bytes": len(manifest_payload),
            "rows": 2,
        },
        "counts": {
            "selected_rows": 2,
            "calibration_rows": 1,
            "evaluation_rows": 1,
        },
        "split_identity_hashes": {
            "calibration_request_ids_sha256": _string_set_sha256(calibration_ids),
            "calibration_prompt_sha256_set_sha256": _string_set_sha256(
                calibration_prompts
            ),
            "evaluation_request_ids_sha256": _string_set_sha256(evaluation_ids),
            "evaluation_prompt_sha256_set_sha256": _string_set_sha256(
                evaluation_prompts
            ),
        },
        "invariants": {
            "exactly_one_decode_position_per_request": True,
            "calibration_evaluation_request_overlap": 0,
            "calibration_evaluation_prompt_overlap": 0,
            "duplicate_selected_request_ids": 0,
            "duplicate_selected_prompt_hashes": 0,
            "model_or_gpu_used": False,
            "terminal_quality_labels_read_or_used": False,
        },
    }
    facts_path = tmp_path / "requests.facts.json"
    facts_sha = _write_json(facts_path, facts)
    return {
        "config": config_path,
        "config_sha": config_sha,
        "manifest": manifest_path,
        "manifest_sha": manifest_sha,
        "request_facts": facts_path,
        "request_facts_sha": facts_sha,
    }


def _candidate_files(
    tmp_path: Path,
    *,
    split: str,
    request_id: str,
    frozen_sha256: str | None = None,
) -> tuple[Path, Path, str]:
    root = tmp_path / "candidates"
    layers: dict[str, Any] = {}
    for layer in LAYERS:
        path = root / split / f"layer_{layer:02d}" / cli.PICKLE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"trusted-{split}-{layer}".encode())
        frozen = (
            None
            if split == "calibration"
            else {
                "frozen_calibration_spec_sha256": frozen_sha256,
                "calibration_evidence_sha256": "e" * 64,
            }
        )
        layers[str(layer)] = {
            "completed": True,
            "schema": cli.CANDIDATE_SCHEMA,
            "split": split,
            "layer": layer,
            "requests": 1,
            "request_ids": [request_id],
            "repair_windows": WINDOWS,
            "eta_grid": ETAS,
            "rates": RATES,
            "rows": 1,
            "expected_rows": 1,
            "allocation_uses_terminal_or_downstream_outcome": False,
            "candidate_execution_stops_at_d1_router": True,
            "frozen_calibration": frozen,
            "files": {
                cli.PICKLE_NAME: {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            },
        }
    run_facts = {
        "completed": True,
        "schema": cli.CANDIDATE_SCHEMA,
        "split": split,
        "layers": layers,
        "runner_sha256": "f" * 64,
        "test_rows_admitted_or_used": False,
    }
    facts_path = root / f"nested_run_facts_{split}.json"
    facts_sha = _write_json(facts_path, run_facts)
    return root, facts_path, facts_sha


def _common_args(
    protocol: dict[str, Any], root: Path, facts_path: Path, facts_sha: str,
) -> list[str]:
    return [
        "--config", str(protocol["config"]),
        "--config-sha256", protocol["config_sha"],
        "--request-manifest", str(protocol["manifest"]),
        "--request-manifest-sha256", protocol["manifest_sha"],
        "--request-manifest-facts", str(protocol["request_facts"]),
        "--request-manifest-facts-sha256", protocol["request_facts_sha"],
        "--candidate-root", str(root),
        "--candidate-run-facts", str(facts_path),
        "--candidate-run-facts-sha256", facts_sha,
    ]


def test_calibrate_cli_discovers_exact_six_pinned_layers_and_forwards_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = _protocol_files(tmp_path)
    root, run_facts, run_sha = _candidate_files(
        tmp_path, split="calibration", request_id="request-calibration",
    )
    observed: dict[str, Any] = {}

    def load(paths, **kwargs):
        observed["paths"] = tuple(paths)
        observed["load"] = kwargs
        return "calibration-corpus"

    def select(corpus, config, **kwargs):
        observed["select"] = (corpus, config, kwargs)
        return "selection"

    def write(selection, **kwargs):
        observed["write"] = (selection, kwargs)
        return {
            "frozen_calibration_spec_sha256": "a" * 64,
            "calibration_evidence_sha256": "b" * 64,
        }

    monkeypatch.setattr(cli, "load_layer_candidate_pickles", load)
    monkeypatch.setattr(cli, "select_and_freeze_calibration", select)
    monkeypatch.setattr(cli, "write_calibration_selection", write)
    frozen = tmp_path / "frozen.json"
    evidence = tmp_path / "evidence.json"
    argv = [
        "calibrate",
        *_common_args(protocol, root, run_facts, run_sha),
        "--frozen-output", str(frozen),
        "--evidence-output", str(evidence),
    ]
    assert cli.main(argv) == 0
    assert observed["paths"] == tuple(
        root / "calibration" / f"layer_{layer:02d}" / cli.PICKLE_NAME
        for layer in LAYERS
    )
    assert observed["load"]["expected_layers"] == LAYERS
    assert len(observed["load"]["expected_file_sha256"]) == 6
    assert observed["select"][2]["expected_request_ids"] == (
        "request-calibration",
    )
    assert observed["select"][2]["config_file_sha256"] == protocol["config_sha"]
    assert observed["write"] == (
        "selection", {"frozen_path": frozen, "evidence_path": evidence},
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "calibrate"
    assert printed["layers"] == list(LAYERS)


def test_seal_evaluation_cli_binds_canonical_frozen_spec_and_run_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = _protocol_files(tmp_path)
    frozen = freeze_calibration_spec({
        "config_file_sha256": protocol["config_sha"],
        "request_manifest_sha256": protocol["manifest_sha"],
        "request_manifest_facts_sha256": protocol["request_facts_sha"],
        "calibration_request_ids": ["request-calibration"],
    })
    frozen_path = tmp_path / "frozen.json"
    frozen_file_sha = _write_json(frozen_path, frozen, canonical=True)
    frozen_digest = frozen["sha256"]
    root, run_facts, run_sha = _candidate_files(
        tmp_path,
        split="evaluation",
        request_id="request-evaluation",
        frozen_sha256=frozen_digest,
    )
    observed: dict[str, Any] = {}

    def load(paths, **kwargs):
        observed["load"] = (tuple(paths), kwargs)
        return "evaluation-corpus"

    def seal(corpus, config, frozen_value, **kwargs):
        observed["seal"] = (corpus, config, frozen_value, kwargs)
        return {"allocation_manifest_sha256": "c" * 64}

    monkeypatch.setattr(cli, "load_layer_candidate_pickles", load)
    monkeypatch.setattr(cli, "write_sealed_evaluation_allocations", seal)
    manifest_output = tmp_path / "allocations.json"
    seal_output = tmp_path / "allocations.seal.json"
    argv = [
        "seal-evaluation",
        *_common_args(protocol, root, run_facts, run_sha),
        "--frozen-spec", str(frozen_path),
        "--frozen-spec-sha256", frozen_file_sha,
        "--manifest-output", str(manifest_output),
        "--seal-output", str(seal_output),
    ]
    assert cli.main(argv) == 0
    assert observed["load"][1]["expected_split"] == "evaluation"
    assert observed["seal"][0] == "evaluation-corpus"
    assert observed["seal"][2] == frozen
    assert observed["seal"][3]["expected_request_ids"] == (
        "request-evaluation",
    )
    assert observed["seal"][3]["manifest_path"] == manifest_output
    assert observed["seal"][3]["seal_path"] == seal_output
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "seal-evaluation"
    assert printed["frozen_calibration_spec_sha256"] == frozen_digest


def test_cli_hash_pins_are_required_and_mismatch_fails_before_pickle_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["calibrate"])
    with pytest.raises(SystemExit):
        cli.parse_args(["seal-evaluation"])

    protocol = _protocol_files(tmp_path)
    root, run_facts, _ = _candidate_files(
        tmp_path, split="calibration", request_id="request-calibration",
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("untrusted pickle loader was reached")

    monkeypatch.setattr(cli, "load_layer_candidate_pickles", forbidden)
    argv = [
        "calibrate",
        *_common_args(protocol, root, run_facts, "0" * 64),
        "--frozen-output", str(tmp_path / "frozen.json"),
        "--evidence-output", str(tmp_path / "evidence.json"),
    ]
    with pytest.raises(SystemExit, match="candidate run facts SHA-256 mismatch"):
        cli.main(argv)
    assert called is False


def test_cross_split_candidate_code_identity_exact_match_passes() -> None:
    identity = {
        "schema": "pr13_d1_nested_code_bundle_v1",
        "files": 17,
        "canonical_sha256": "d" * 64,
    }
    frozen = {
        "spec": {"calibration_candidate_code_identity": identity},
    }
    cli._require_matching_candidate_code_identity(frozen, dict(identity))


def test_seal_rejects_code_identity_mismatch_before_pickle_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol_files(tmp_path)
    calibration_identity = {
        "schema": "pr13_d1_nested_code_bundle_v1",
        "files": 17,
        "canonical_sha256": "d" * 64,
    }
    evaluation_identity = {
        **calibration_identity,
        "canonical_sha256": "e" * 64,
    }
    frozen = freeze_calibration_spec({
        "config_file_sha256": protocol["config_sha"],
        "request_manifest_sha256": protocol["manifest_sha"],
        "request_manifest_facts_sha256": protocol["request_facts_sha"],
        "calibration_request_ids": ["request-calibration"],
        "calibration_candidate_code_identity": calibration_identity,
    })
    frozen_path = tmp_path / "frozen-code-identity.json"
    frozen_file_sha = _write_json(frozen_path, frozen, canonical=True)
    root, run_facts, run_sha = _candidate_files(
        tmp_path,
        split="evaluation",
        request_id="request-evaluation",
        frozen_sha256=frozen["sha256"],
    )
    monkeypatch.setattr(
        cli,
        "_discover_candidate_inputs",
        lambda *args, **kwargs: cli.CandidateInputs(
            (), {}, run_sha, evaluation_identity,
        ),
    )
    pickle_loaded = False

    def forbidden(*args, **kwargs):
        nonlocal pickle_loaded
        pickle_loaded = True
        raise AssertionError("mismatched-code pickle loader was reached")

    monkeypatch.setattr(cli, "load_layer_candidate_pickles", forbidden)
    args = cli.parse_args([
        "seal-evaluation",
        *_common_args(protocol, root, run_facts, run_sha),
        "--frozen-spec", str(frozen_path),
        "--frozen-spec-sha256", frozen_file_sha,
        "--manifest-output", str(tmp_path / "allocations.json"),
        "--seal-output", str(tmp_path / "allocations.seal.json"),
    ])
    with pytest.raises(
        ArtifactValidationError,
        match="code identity differs from frozen calibration",
    ):
        cli.run_seal_evaluation(args)
    assert pickle_loaded is False
