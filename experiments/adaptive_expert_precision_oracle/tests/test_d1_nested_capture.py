from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "scripts"), str(EXPERIMENT / "src")]

import capture_d1_nested_exact_decode as capture  # noqa: E402


DOMAINS = tuple(f"domain-{index}" for index in range(8))
INJECTION_LAYERS = (0, 2)
ROUTER_LAYERS = (0, 1, 2, 3)
MODEL_LAYERS = 4
HIDDEN_SIZE = 3
NUM_EXPERTS = 10
INPUT_HASHES = {
    "config_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
    "manifest_facts_sha256": "c" * 64,
}
REPEAT = {
    "hidden_max_abs": 0.0,
    "router_logits_max_abs": 0.0,
    "router_scores_max_abs": 0.0,
    "x_max_abs": 0.0,
    "residual_max_abs": 0.0,
    "routed_max_abs": 0.0,
    "router_ids_equal": True,
    "post_token_cache_tensors": 4,
    "post_token_cache_coordinates": 32,
    "post_token_cache_bit_identical": True,
    "post_token_cache_max_abs": 0.0,
    "post_token_cache_mse": 0.0,
    "all_exact": True,
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config() -> dict[str, object]:
    return {
        "schema": capture.CONFIG_SCHEMA,
        "experiment_stage": "A",
        "checkpoint_config_sha256": "d" * 64,
        "checkpoint_index_sha256": "e" * 64,
        "request_source": {
            "path": "/absent/source.jsonl",
            "sha256": "f" * 64,
            "selected_manifest": "/absent/selected.jsonl",
            "selected_manifest_facts": "/absent/selected_facts.json",
            "domains": list(DOMAINS),
            "selection_seed": 20260825,
            "calibration_requests_per_domain": 4,
            "evaluation_requests_per_domain": 12,
            "total_requests": 128,
            "development_prompt_sha256_exclusions": ["1" * 64, "2" * 64, "3" * 64],
        },
        "decode_position": {
            "maximum_position": 3,
            "minimum_prefix_tokens": 1,
        },
    }


def _rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    uint64_max = (1 << 64) - 1
    for domain_index, domain in enumerate(DOMAINS):
        for rank in range(16):
            request_id = str(uint64_max - domain_index * 16 - rank)
            prompt_sha = hashlib.sha256(f"{domain}:{rank}".encode()).hexdigest()
            tokens = [domain_index, rank, 100, 101, 102]
            split = "calibration" if rank < 4 else "evaluation"
            result.append({
                "schema": capture.REQUEST_MANIFEST_SCHEMA,
                "request_id": request_id,
                "domain": domain,
                "prompt_sha256": prompt_sha,
                "prompt_token_ids": tokens,
                "decode_position": 3,
                "prefix_tokens": 3,
                "next_token_id": 102,
                "selection_hash": capture._selection_hash(
                    seed=20260825, domain=domain, prompt_sha256=prompt_sha,
                    request_id=request_id,
                ),
                "split": split,
                "domain_rank": rank,
                "split_rank": rank if split == "calibration" else rank - 4,
            })
    return result


def _write_input_bundle(
    root: Path,
) -> tuple[Path, Path, Path, Path, list[dict[str, object]]]:
    config = _config()
    reference_config = root / "reference" / "config.json"
    staged_config = root / "staged" / "renamed-config.json"
    reference_config.parent.mkdir(parents=True)
    staged_config.parent.mkdir(parents=True)
    config_payload = (json.dumps(config, sort_keys=True) + "\n").encode()
    reference_config.write_bytes(config_payload)
    staged_config.write_bytes(config_payload)
    rows = _rows()
    manifest = root / "elsewhere" / "requests.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest_payload = "".join(
        capture.canonical_json(row) + "\n" for row in rows
    ).encode()
    manifest.write_bytes(manifest_payload)
    calibration = [row for row in rows if row["split"] == "calibration"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    facts = {
        "schema": capture.REQUEST_MANIFEST_FACTS_SCHEMA,
        "completed": True,
        "source": {"sha256": config["request_source"]["sha256"]},
        "selection": {
            "schema": capture.REQUEST_MANIFEST_SCHEMA,
            "seed": 20260825,
            "domains": sorted(DOMAINS),
            "required_domain_count": 8,
            "calibration_per_domain": 4,
            "evaluation_per_domain": 12,
            "max_position": 3,
            "min_prefix_tokens": 1,
            "excluded_prompt_sha256": ["1" * 64, "2" * 64, "3" * 64],
            "rank_algorithm": (
                "sha256(canonical_json([schema,seed,domain,prompt_sha256,request_id]))"
            ),
        },
        "output": {
            "path": "/different/absolute/path.jsonl",
            "bytes": len(manifest_payload),
            "rows": len(rows),
            "sha256": _sha(manifest_payload),
        },
        "split_identity_hashes": {
            "calibration_request_ids_sha256": capture._sha256_string_set(
                [str(row["request_id"]) for row in calibration]
            ),
            "calibration_prompt_sha256_set_sha256": capture._sha256_string_set(
                [str(row["prompt_sha256"]) for row in calibration]
            ),
            "evaluation_request_ids_sha256": capture._sha256_string_set(
                [str(row["request_id"]) for row in evaluation]
            ),
            "evaluation_prompt_sha256_set_sha256": capture._sha256_string_set(
                [str(row["prompt_sha256"]) for row in evaluation]
            ),
        },
    }
    facts_path = root / "also-elsewhere" / "manifest-facts.json"
    facts_path.parent.mkdir(parents=True)
    facts_path.write_text(json.dumps(facts, sort_keys=True))
    return staged_config, reference_config, manifest, facts_path, rows


def _observation(index: int) -> capture.TokenObservation:
    hidden = {
        layer: torch.full((1, 1, HIDDEN_SIZE), float(index + layer))
        for layer in INJECTION_LAYERS
    }
    router_logits = {
        layer: torch.arange(NUM_EXPERTS, dtype=torch.float32).reshape(1, 1, -1)
        + index + layer
        for layer in ROUTER_LAYERS
    }
    router_scores = {
        layer: torch.full((1, 1, 8), 0.125, dtype=torch.float32)
        for layer in ROUTER_LAYERS
    }
    router_ids = {
        layer: torch.arange(8, dtype=torch.long).reshape(1, 1, -1)
        for layer in ROUTER_LAYERS
    }
    local = {
        layer: torch.full((1, 1, HIDDEN_SIZE), float(index + layer + 1))
        for layer in INJECTION_LAYERS
    }
    return capture.TokenObservation(
        hidden=hidden, router_logits=router_logits,
        router_scores=router_scores, router_ids=router_ids,
        x=local, residual=deepcopy(local), routed=deepcopy(local),
        cache=None, elapsed_seconds=0.1,
    )


def _archive_arrays(row: dict[str, object]) -> dict[str, np.ndarray]:
    return capture._capture_arrays(
        row, [_observation(index) for index in range(4)],
        INJECTION_LAYERS, ROUTER_LAYERS, model_layers=MODEL_LAYERS,
        input_hashes=INPUT_HASHES, repeat_facts=REPEAT,
    )


def _trusted_record(
    path: Path,
    row: dict[str, object],
    repeat: dict[str, object] = REPEAT,
) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": capture.sha256(path),
        "request_manifest_row_sha256": capture.canonical_sha256(row),
        "repeat_gate_sha256": capture.canonical_sha256(repeat),
    }


def test_input_bundle_is_hash_pinned_but_path_portable(tmp_path: Path) -> None:
    config, frozen, manifest, facts, rows = _write_input_bundle(tmp_path)
    loaded_config, loaded_rows, identity = capture.validate_capture_inputs(
        config_path=config, manifest_path=manifest, manifest_facts_path=facts,
        frozen_config_path=frozen,
    )
    assert loaded_config["schema"] == capture.CONFIG_SCHEMA
    assert loaded_rows == rows
    assert identity["config_sha256"] == capture.sha256(config)
    assert identity["manifest_sha256"] == capture.sha256(manifest)
    assert capture.capture_path(tmp_path, str((1 << 64) - 1)).name == (
        "18446744073709551615.npz"
    )


def test_prefix_progresses_once_and_drops_historical_cache_references() -> None:
    class Observer:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, object | None]] = []
            self.caches: list[object] = []

        def run(
            self, _model: object, token: int, position: int,
            prefix: object | None, _device: torch.device,
        ) -> SimpleNamespace:
            self.calls.append((token, position, prefix))
            cache = object()
            self.caches.append(cache)
            return SimpleNamespace(cache=cache, elapsed_seconds=0.25)

    observer = Observer()
    observations, prefix, elapsed = capture._advance_exact_prefix(
        observer, object(), [10, 11, 12, 13], 3, torch.device("cpu"),
    )
    assert [(token, position) for token, position, _ in observer.calls] == [
        (10, 0), (11, 1), (12, 2),
    ]
    assert observer.calls[0][2] is None
    assert observer.calls[1][2] is observer.caches[0]
    assert observer.calls[2][2] is observer.caches[1]
    assert prefix is observer.caches[2]
    assert all(observation.cache is None for observation in observations)
    assert elapsed == 0.75


def test_archive_shapes_are_consumer_compatible_and_outcome_free(tmp_path: Path) -> None:
    row = _rows()[0]
    arrays = _archive_arrays(row)
    assert arrays["hidden"].shape == (MODEL_LAYERS, 4, HIDDEN_SIZE)
    assert arrays["router_logits"].shape == (MODEL_LAYERS, 4, NUM_EXPERTS)
    assert arrays["router_scores"].shape == (MODEL_LAYERS, 4, 8)
    assert arrays["router_ids"].shape == (MODEL_LAYERS, 4, 8)
    assert np.isnan(arrays["hidden"][1]).all()
    assert not any("terminal" in name or "downstream" in name for name in arrays)
    path = tmp_path / "capture.npz"
    capture.atomic_npz(path, **arrays)
    assert capture.validate_capture_archive(
        path, row, input_hashes=INPUT_HASHES,
        injection_layers=INJECTION_LAYERS, router_layers=ROUTER_LAYERS,
        model_layers=MODEL_LAYERS, hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
    ) == REPEAT
    assert capture._validate_existing(
        path, row, trusted_record=_trusted_record(path, row),
        input_hashes=INPUT_HASHES, injection_layers=INJECTION_LAYERS,
        router_layers=ROUTER_LAYERS, model_layers=MODEL_LAYERS,
        hidden_size=HIDDEN_SIZE, num_experts=NUM_EXPERTS,
    ) == REPEAT


def test_resume_rejects_hash_shape_and_repeat_gate_tampering(tmp_path: Path) -> None:
    row = _rows()[0]
    good = tmp_path / "good.npz"
    capture.atomic_npz(good, **_archive_arrays(row))
    bad_hash = _trusted_record(good, row)
    bad_hash["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="hash/row identity"):
        capture._validate_existing(
            good, row, trusted_record=bad_hash, input_hashes=INPUT_HASHES,
            injection_layers=INJECTION_LAYERS, router_layers=ROUTER_LAYERS,
            model_layers=MODEL_LAYERS, hidden_size=HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
        )

    shape_arrays = _archive_arrays(row)
    shape_arrays["hidden"] = shape_arrays["hidden"][:-1]
    bad_shape = tmp_path / "bad-shape.npz"
    capture.atomic_npz(bad_shape, **shape_arrays)
    with pytest.raises(RuntimeError, match="hidden array shape"):
        capture._validate_existing(
            bad_shape, row, trusted_record=_trusted_record(bad_shape, row),
            input_hashes=INPUT_HASHES, injection_layers=INJECTION_LAYERS,
            router_layers=ROUTER_LAYERS, model_layers=MODEL_LAYERS,
            hidden_size=HIDDEN_SIZE, num_experts=NUM_EXPERTS,
        )

    failed_repeat = dict(REPEAT)
    failed_repeat["all_exact"] = False
    repeat_arrays = capture._capture_arrays(
        row, [_observation(index) for index in range(4)],
        INJECTION_LAYERS, ROUTER_LAYERS, model_layers=MODEL_LAYERS,
        input_hashes=INPUT_HASHES, repeat_facts=failed_repeat,
    )
    bad_repeat = tmp_path / "bad-repeat.npz"
    capture.atomic_npz(bad_repeat, **repeat_arrays)
    with pytest.raises(RuntimeError, match="exactness gate failed"):
        capture._validate_existing(
            bad_repeat, row,
            trusted_record=_trusted_record(bad_repeat, row, failed_repeat),
            input_hashes=INPUT_HASHES, injection_layers=INJECTION_LAYERS,
            router_layers=ROUTER_LAYERS, model_layers=MODEL_LAYERS,
            hidden_size=HIDDEN_SIZE, num_experts=NUM_EXPERTS,
        )
