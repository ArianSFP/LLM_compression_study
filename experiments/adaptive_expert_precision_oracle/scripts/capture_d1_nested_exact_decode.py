#!/usr/bin/env python3
"""Capture independent exact-prefix, single-token inputs for nested D1 Experiment A.

The model is advanced token by token through the exact prompt prefix.  Only
current/intermediate tensors needed to construct PR13 states and the same-token
next-layer D1 label are serialized.  Terminal logits, downstream candidate
routes, and quality outcomes are deliberately neither retained nor written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.d1_decode import cache_state_metrics, clone_decode_cache  # noqa: E402
from run_same_host_causal_controls import _load_model  # noqa: E402


SCHEMA = "pr13_d1_nested_exact_decode_capture_v1"
FACTS_SCHEMA = "pr13_d1_nested_exact_decode_capture_facts_v1"
REQUEST_MANIFEST_SCHEMA = "pr13_d1_nested_request_manifest_v1"
REQUEST_MANIFEST_FACTS_SCHEMA = "pr13_d1_nested_request_manifest_facts_v1"
CONFIG_SCHEMA = "pr13_d1_nested_safe_oracle_config_v1"
FROZEN_CONFIG_PATH = (
    EXPERIMENT / "configs" / "qwen36_mxfp4_d1_nested_safe_oracle_20260825_v1.json"
)
TOP_K = 8
REQUEST_ROW_FIELDS = frozenset({
    "schema", "request_id", "domain", "prompt_sha256", "prompt_token_ids",
    "decode_position", "prefix_tokens", "next_token_id", "selection_hash",
    "split", "domain_rank", "split_rank",
})
BASE_ARRAY_FIELDS = frozenset({
    "schema", "request_id", "domain", "split", "prompt_sha256",
    "request_manifest_row_sha256", "config_sha256", "manifest_sha256",
    "manifest_facts_sha256", "decode_position", "next_token_id", "input_ids",
    "attention_mask", "hidden", "router_logits", "router_scores", "router_ids",
    "current_token_repeat_facts_json", "current_token_repeat_facts_sha256",
    "current_token_repeat_all_exact",
})
REPEAT_FLOAT_ZERO_FIELDS = frozenset({
    "hidden_max_abs", "router_logits_max_abs", "router_scores_max_abs",
    "x_max_abs", "residual_max_abs", "routed_max_abs",
    "post_token_cache_max_abs", "post_token_cache_mse",
})
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]+\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    )


def _json_no_duplicate_keys(payload: str, *, source: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    return json.loads(payload, object_pairs_hook=pairs_hook)


def load_json(path: Path) -> dict[str, Any]:
    value = _json_no_duplicate_keys(path.read_text(), source=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            raise ValueError(f"blank request-manifest row at line {line_number}")
        row = _json_no_duplicate_keys(
            line, source=f"{path} line {line_number}",
        )
        if not isinstance(row, dict):
            raise ValueError("request manifest must contain JSON objects")
        rows.append(row)
    if not rows:
        raise ValueError("request manifest is empty")
    return rows


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def atomic_json(path: Path, payload: Any) -> None:
    temporary: Path | None = None
    try:
        temporary = _stage_bytes(
            path,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
        )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_path(directory: Path, request_id: str) -> Path:
    safe = str(request_id)
    if (
        not safe or safe in {".", ".."} or SAFE_REQUEST_ID.fullmatch(safe) is None
        or len(safe.encode("utf-8")) > 240
    ):
        raise ValueError("request_id is not a safe capture filename")
    return directory / f"{safe}.npz"


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _selection_hash(
    *, seed: int, domain: str, prompt_sha256: str, request_id: str,
) -> str:
    identity = json.dumps(
        [REQUEST_MANIFEST_SCHEMA, seed, domain, prompt_sha256, request_id],
        ensure_ascii=False, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(identity).hexdigest()


def _sha256_string_set(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(values))).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_request_manifest_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate every selected request against the frozen, label-free rule."""

    request_source = config["request_source"]
    decode = config["decode_position"]
    domains = tuple(map(str, request_source["domains"]))
    if len(domains) != 8 or len(set(domains)) != 8:
        raise ValueError("frozen request domain grid is not exactly eight unique domains")
    calibration_per_domain = int(request_source["calibration_requests_per_domain"])
    evaluation_per_domain = int(request_source["evaluation_requests_per_domain"])
    selected_per_domain = calibration_per_domain + evaluation_per_domain
    expected_rows = int(request_source["total_requests"])
    if len(rows) != expected_rows:
        raise ValueError(f"request manifest row count changed: {len(rows)} != {expected_rows}")
    maximum_position = int(decode["maximum_position"])
    minimum_prefix = int(decode["minimum_prefix_tokens"])
    seed = int(request_source["selection_seed"])
    seen_requests: set[str] = set()
    seen_prompts: set[str] = set()
    domain_ranks: dict[str, set[int]] = {domain: set() for domain in domains}
    split_counts: dict[tuple[str, str], int] = {}
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        if set(row) != REQUEST_ROW_FIELDS:
            raise ValueError(f"request manifest row {index} field schema changed")
        if row["schema"] != REQUEST_MANIFEST_SCHEMA:
            raise ValueError(f"request manifest row {index} schema changed")
        request_id = row["request_id"]
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"request manifest row {index} request_id is not normalized")
        capture_path(Path("."), request_id)
        domain = row["domain"]
        if domain not in domain_ranks:
            raise ValueError(f"request manifest row {index} has an unknown domain")
        prompt_sha = row["prompt_sha256"]
        if not _valid_sha256(prompt_sha):
            raise ValueError(f"request manifest row {index} prompt hash is invalid")
        tokens = row["prompt_token_ids"]
        if (
            not isinstance(tokens, list)
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            )
        ):
            raise ValueError(f"request manifest row {index} token IDs are invalid")
        position = row["decode_position"]
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(f"request manifest row {index} decode position is invalid")
        expected_position = min(maximum_position, len(tokens) - 2)
        if position != expected_position or position < minimum_prefix:
            raise ValueError(f"request manifest row {index} decode position changed")
        if row["prefix_tokens"] != position or row["next_token_id"] != tokens[position + 1]:
            raise ValueError(f"request manifest row {index} token boundary changed")
        rank = row["domain_rank"]
        split_rank = row["split_rank"]
        if (
            isinstance(rank, bool) or not isinstance(rank, int)
            or not 0 <= rank < selected_per_domain
        ):
            raise ValueError(f"request manifest row {index} domain rank changed")
        expected_split = "calibration" if rank < calibration_per_domain else "evaluation"
        expected_split_rank = (
            rank if expected_split == "calibration"
            else rank - calibration_per_domain
        )
        if row["split"] != expected_split or split_rank != expected_split_rank:
            raise ValueError(f"request manifest row {index} split assignment changed")
        expected_hash = _selection_hash(
            seed=seed, domain=domain, prompt_sha256=prompt_sha,
            request_id=request_id,
        )
        if row["selection_hash"] != expected_hash:
            raise ValueError(f"request manifest row {index} selection hash changed")
        if request_id in seen_requests or prompt_sha in seen_prompts:
            raise ValueError("request manifest contains duplicate request or prompt identity")
        seen_requests.add(request_id)
        seen_prompts.add(prompt_sha)
        if rank in domain_ranks[domain]:
            raise ValueError(f"request manifest domain {domain!r} repeats a rank")
        domain_ranks[domain].add(rank)
        split_counts[(domain, expected_split)] = split_counts.get((domain, expected_split), 0) + 1
        validated.append(row)
    for domain in domains:
        if domain_ranks[domain] != set(range(selected_per_domain)):
            raise ValueError(f"request manifest domain {domain!r} rank grid changed")
        if split_counts.get((domain, "calibration"), 0) != calibration_per_domain:
            raise ValueError(f"request manifest domain {domain!r} calibration count changed")
        if split_counts.get((domain, "evaluation"), 0) != evaluation_per_domain:
            raise ValueError(f"request manifest domain {domain!r} evaluation count changed")
    return validated


def _resolve_manifest_facts_path(
    config: Mapping[str, Any], manifest_path: Path, supplied: Path | None,
) -> Path:
    if supplied is not None:
        return Path(supplied)
    sibling = manifest_path.with_name(f"{manifest_path.stem}_facts.json")
    if sibling.is_file():
        return sibling
    configured = Path(str(config["request_source"]["selected_manifest_facts"]))
    if configured.is_file():
        return configured
    raise FileNotFoundError(
        "selected request-manifest facts are required; supply --manifest-facts"
    )


def validate_capture_inputs(
    *, config_path: Path, manifest_path: Path,
    manifest_facts_path: Path | None = None,
    frozen_config_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Byte-pin staged inputs by digest, never by absolute-path equality."""

    reference_config = Path(frozen_config_path or FROZEN_CONFIG_PATH)
    config_sha = sha256(config_path)
    if config_sha != sha256(reference_config):
        raise ValueError("supplied config bytes differ from the frozen checked-in config")
    config = load_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("experiment_stage") != "A":
        raise ValueError("capture requires the frozen nested Experiment A config")
    manifest_sha = sha256(manifest_path)
    rows = validate_request_manifest_rows(load_jsonl(manifest_path), config)
    facts_path = _resolve_manifest_facts_path(config, manifest_path, manifest_facts_path)
    manifest_facts = load_json(facts_path)
    if (
        manifest_facts.get("schema") != REQUEST_MANIFEST_FACTS_SCHEMA
        or manifest_facts.get("completed") is not True
    ):
        raise ValueError("request-manifest facts are incomplete or use the wrong schema")
    output = manifest_facts.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("request-manifest facts output identity is absent")
    if (
        output.get("sha256") != manifest_sha
        or int(output.get("bytes", -1)) != manifest_path.stat().st_size
        or int(output.get("rows", -1)) != len(rows)
    ):
        raise ValueError("supplied request manifest differs from its pinned facts")
    source = manifest_facts.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != config["request_source"]["sha256"]
    ):
        raise ValueError("request-manifest facts do not derive from the frozen source")
    selection = manifest_facts.get("selection")
    expected_selection = {
        "schema": REQUEST_MANIFEST_SCHEMA,
        "seed": int(config["request_source"]["selection_seed"]),
        "domains": sorted(map(str, config["request_source"]["domains"])),
        "required_domain_count": 8,
        "calibration_per_domain": int(config["request_source"]["calibration_requests_per_domain"]),
        "evaluation_per_domain": int(config["request_source"]["evaluation_requests_per_domain"]),
        "max_position": int(config["decode_position"]["maximum_position"]),
        "min_prefix_tokens": int(config["decode_position"]["minimum_prefix_tokens"]),
        "excluded_prompt_sha256": sorted(map(
            str, config["request_source"]["development_prompt_sha256_exclusions"],
        )),
        "rank_algorithm": "sha256(canonical_json([schema,seed,domain,prompt_sha256,request_id]))",
    }
    if selection != expected_selection:
        raise ValueError("request-manifest selection spec differs from the frozen config")
    split_hashes = manifest_facts.get("split_identity_hashes")
    calibration = [row for row in rows if row["split"] == "calibration"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    expected_split_hashes = {
        "calibration_request_ids_sha256": _sha256_string_set(
            [row["request_id"] for row in calibration]
        ),
        "calibration_prompt_sha256_set_sha256": _sha256_string_set(
            [row["prompt_sha256"] for row in calibration]
        ),
        "evaluation_request_ids_sha256": _sha256_string_set(
            [row["request_id"] for row in evaluation]
        ),
        "evaluation_prompt_sha256_set_sha256": _sha256_string_set(
            [row["prompt_sha256"] for row in evaluation]
        ),
    }
    if split_hashes != expected_split_hashes:
        raise ValueError("request-manifest split identities differ from pinned facts")
    return config, rows, {
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "manifest_facts_path": str(facts_path),
        "manifest_facts_sha256": sha256(facts_path),
    }


def validate_checkpoint_pins(config: Mapping[str, Any], checkpoint: Path) -> dict[str, str]:
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    observed = {
        "checkpoint_config_sha256": sha256(config_path),
        "checkpoint_index_sha256": sha256(index_path),
    }
    if observed["checkpoint_config_sha256"] != config["checkpoint_config_sha256"]:
        raise RuntimeError("checkpoint config differs from the frozen config")
    if observed["checkpoint_index_sha256"] != config["checkpoint_index_sha256"]:
        raise RuntimeError("checkpoint index differs from the frozen config")
    return observed


@dataclass
class TokenObservation:
    hidden: dict[int, torch.Tensor]
    router_logits: dict[int, torch.Tensor]
    router_scores: dict[int, torch.Tensor]
    router_ids: dict[int, torch.Tensor]
    x: dict[int, torch.Tensor]
    residual: dict[int, torch.Tensor]
    routed: dict[int, torch.Tensor]
    cache: Any
    elapsed_seconds: float


class IndependentDecodeCaptureObserver:
    """Capture the minimal baseline tensors for six D1 injection layers."""

    def __init__(self, model: Any, injection_layers: Sequence[int]) -> None:
        self.injection_layers = tuple(map(int, injection_layers))
        self.router_layers = tuple(sorted(set(
            self.injection_layers + tuple(layer + 1 for layer in self.injection_layers)
        )))
        self.hidden_layers = self.injection_layers
        self.handles: list[Any] = []
        layers = model.model.language_model.layers
        self.model_layers = len(layers)
        if any(layer < 0 or layer + 1 >= len(layers) for layer in self.injection_layers):
            raise ValueError("every injection layer must have a D1 successor")
        for layer_id in self.hidden_layers:
            layer = layers[layer_id]

            def layer_hook(
                _module: Any, _inputs: Any, output: Any, layer_id: int = layer_id,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                self.hidden[layer_id] = value.detach().cpu().clone()

            def norm_hook(
                _module: Any,
                inputs: Any,
                output: torch.Tensor,
                layer_id: int = layer_id,
            ) -> None:
                self.residual[layer_id] = inputs[0].detach().cpu().clone()
                self.x[layer_id] = output.detach().cpu().clone()

            def routed_hook(
                _module: Any,
                _inputs: Any,
                output: torch.Tensor,
                layer_id: int = layer_id,
            ) -> None:
                self.routed[layer_id] = output.detach().cpu().clone()

            self.handles.append(layer.register_forward_hook(layer_hook))
            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(norm_hook)
            )
            self.handles.append(layer.mlp.experts.register_forward_hook(routed_hook))
        for layer_id in self.router_layers:
            gate = layers[layer_id].mlp.gate

            def gate_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer_id: int = layer_id,
            ) -> None:
                if not isinstance(output, tuple) or len(output) < 3:
                    raise RuntimeError("router output contract changed")
                self.router_logits[layer_id] = output[0].detach().cpu().clone()
                self.router_scores[layer_id] = output[1].detach().cpu().clone()
                self.router_ids[layer_id] = output[2].detach().cpu().clone()

            self.handles.append(gate.register_forward_hook(gate_hook))
        self.reset()

    def reset(self) -> None:
        self.hidden: dict[int, torch.Tensor] = {}
        self.router_logits: dict[int, torch.Tensor] = {}
        self.router_scores: dict[int, torch.Tensor] = {}
        self.router_ids: dict[int, torch.Tensor] = {}
        self.x: dict[int, torch.Tensor] = {}
        self.residual: dict[int, torch.Tensor] = {}
        self.routed: dict[int, torch.Tensor] = {}

    def run(
        self,
        model: Any,
        token_id: int,
        position: int,
        prefix_cache: Any | None,
        device: torch.device,
    ) -> TokenObservation:
        self.reset()
        position = int(position)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=torch.as_tensor([[int(token_id)]], device=device),
                attention_mask=torch.ones(
                    (1, position + 1), device=device, dtype=torch.long,
                ),
                past_key_values=prefix_cache,
                cache_position=torch.as_tensor(
                    [position], device=device, dtype=torch.long,
                ),
                use_cache=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
        for name, expected, observed in (
            ("hidden", set(self.hidden_layers), set(self.hidden)),
            ("router_logits", set(self.router_layers), set(self.router_logits)),
            ("router_scores", set(self.router_layers), set(self.router_scores)),
            ("router_ids", set(self.router_layers), set(self.router_ids)),
            ("x", set(self.injection_layers), set(self.x)),
            ("residual", set(self.injection_layers), set(self.residual)),
            ("routed", set(self.injection_layers), set(self.routed)),
        ):
            if observed != expected:
                raise RuntimeError(f"incomplete exact decode capture for {name}")
        result = TokenObservation(
            hidden=dict(self.hidden),
            router_logits=dict(self.router_logits),
            router_scores=dict(self.router_scores),
            router_ids=dict(self.router_ids),
            x=dict(self.x),
            residual=dict(self.residual),
            routed=dict(self.routed),
            cache=output.past_key_values,
            elapsed_seconds=time.perf_counter() - started,
        )
        del output
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _flat(value: torch.Tensor, *, integer: bool = False) -> np.ndarray:
    if integer:
        return value.detach().cpu().reshape(-1).numpy().astype(np.int64)
    return value.detach().float().cpu().reshape(-1).numpy().astype(np.float32)


def _advance_exact_prefix(
    observer: Any,
    model: Any,
    tokens: Sequence[int],
    position: int,
    device: torch.device,
) -> tuple[list[TokenObservation], Any, float]:
    """Advance positions ``[0, position)`` once without retaining cache history."""

    observations: list[TokenObservation] = []
    prefix: Any | None = None
    elapsed = 0.0
    for token_position in range(int(position)):
        observation = observer.run(
            model, int(tokens[token_position]), token_position, prefix, device,
        )
        elapsed += float(observation.elapsed_seconds)
        prefix = observation.cache
        observation.cache = None
        observations.append(observation)
    if prefix is None:
        raise RuntimeError("exact decode capture requires a non-empty prefix")
    return observations, prefix, elapsed


def _repeat_exact(first: TokenObservation, second: TokenObservation) -> dict[str, Any]:
    maxima: dict[str, float] = {}
    equal = True
    for name in ("hidden", "router_logits", "router_scores", "x", "residual", "routed"):
        left = getattr(first, name)
        right = getattr(second, name)
        maximum = max(
            float(torch.max(torch.abs(left[key].float() - right[key].float())).item())
            for key in left
        )
        maxima[f"{name}_max_abs"] = maximum
        equal = equal and maximum == 0.0
    ids_equal = all(
        torch.equal(first.router_ids[layer], second.router_ids[layer])
        for layer in first.router_ids
    )
    cache = cache_state_metrics(first.cache, second.cache)
    return {
        **maxima,
        "router_ids_equal": bool(ids_equal),
        **cache.to_dict("post_token_cache"),
        "all_exact": bool(equal and ids_equal and cache.bit_identical),
    }


def _capture_arrays(
    row: Mapping[str, Any],
    observations: Sequence[TokenObservation],
    injection_layers: Sequence[int],
    router_layers: Sequence[int],
    *,
    model_layers: int,
    input_hashes: Mapping[str, str],
    repeat_facts: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    tokens = np.asarray(row["prompt_token_ids"][: int(row["decode_position"]) + 1], np.int64)
    sequence = len(tokens)
    hidden_size = int(next(iter(observations[0].hidden.values())).shape[-1])
    num_layers = int(model_layers)
    if num_layers <= max(max(injection_layers), max(router_layers)):
        raise ValueError("capture layer grid does not fit the full model")
    num_experts = int(next(iter(observations[0].router_logits.values())).shape[-1])
    repeat_json = canonical_json(dict(repeat_facts))
    hidden = np.full((num_layers, sequence, hidden_size), np.nan, np.float32)
    router_logits = np.full((num_layers, sequence, num_experts), np.nan, np.float32)
    router_scores = np.full((num_layers, sequence, 8), np.nan, np.float32)
    router_ids = np.full((num_layers, sequence, 8), -1, np.int64)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "request_id": np.asarray(str(row["request_id"])),
        "domain": np.asarray(str(row["domain"])),
        "split": np.asarray(str(row["split"])),
        "prompt_sha256": np.asarray(str(row["prompt_sha256"])),
        "request_manifest_row_sha256": np.asarray(canonical_sha256(dict(row))),
        "config_sha256": np.asarray(str(input_hashes["config_sha256"])),
        "manifest_sha256": np.asarray(str(input_hashes["manifest_sha256"])),
        "manifest_facts_sha256": np.asarray(str(input_hashes["manifest_facts_sha256"])),
        "decode_position": np.asarray(int(row["decode_position"]), np.int64),
        "next_token_id": np.asarray(int(row["next_token_id"]), np.int64),
        "input_ids": tokens,
        "attention_mask": np.ones(sequence, np.int64),
        "current_token_repeat_facts_json": np.asarray(repeat_json),
        "current_token_repeat_facts_sha256": np.asarray(
            hashlib.sha256(repeat_json.encode()).hexdigest(),
        ),
        "current_token_repeat_all_exact": np.asarray(
            bool(repeat_facts.get("all_exact")), np.bool_,
        ),
    }
    for position, observation in enumerate(observations):
        for layer in injection_layers:
            hidden[layer, position] = _flat(observation.hidden[layer])
        for layer in router_layers:
            router_logits[layer, position] = _flat(observation.router_logits[layer])
            router_scores[layer, position] = _flat(observation.router_scores[layer])
            router_ids[layer, position] = _flat(
                observation.router_ids[layer], integer=True,
            )
    arrays.update({
        "hidden": hidden,
        "router_logits": router_logits,
        "router_scores": router_scores,
        "router_ids": router_ids,
    })
    for layer in injection_layers:
        arrays[f"residual_layer_{layer:02d}"] = np.stack([
            _flat(observation.residual[layer]) for observation in observations
        ]).astype(np.float32)
        arrays[f"x_layer_{layer:02d}"] = np.stack([
            _flat(observation.x[layer]) for observation in observations
        ]).astype(np.float32)
        arrays[f"routed_layer_{layer:02d}"] = np.stack([
            _flat(observation.routed[layer]) for observation in observations
        ]).astype(np.float32)
    return arrays


def _scalar(source: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(source[name])
    if value.shape != ():
        raise RuntimeError(f"capture scalar {name} changed shape")
    return value.item()


def _validate_repeat_facts(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("current-token repeat facts are not an object")
    repeat = dict(value)
    required = set(REPEAT_FLOAT_ZERO_FIELDS) | {
        "router_ids_equal", "post_token_cache_tensors",
        "post_token_cache_coordinates", "post_token_cache_bit_identical",
        "all_exact",
    }
    if set(repeat) != required:
        raise RuntimeError("current-token repeat fact schema changed")
    if any(float(repeat[name]) != 0.0 for name in REPEAT_FLOAT_ZERO_FIELDS):
        raise RuntimeError("current-token repeat fact contains nonzero drift")
    if (
        repeat["router_ids_equal"] is not True
        or repeat["post_token_cache_bit_identical"] is not True
        or repeat["all_exact"] is not True
    ):
        raise RuntimeError("current-token repeat exactness gate failed")
    for name in ("post_token_cache_tensors", "post_token_cache_coordinates"):
        count = repeat[name]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RuntimeError(f"current-token repeat {name} is invalid")
    return repeat


def _expected_array_fields(injection_layers: Sequence[int]) -> set[str]:
    result = set(BASE_ARRAY_FIELDS)
    for layer in injection_layers:
        for prefix in ("residual", "x", "routed"):
            result.add(f"{prefix}_layer_{int(layer):02d}")
    return result


def _validate_sparse_float_layers(
    value: np.ndarray, *, active_layers: set[int], name: str,
) -> None:
    for layer in range(value.shape[0]):
        active = layer in active_layers
        if active and not np.isfinite(value[layer]).all():
            raise RuntimeError(f"capture {name} active layer {layer} is not finite")
        if not active and not np.isnan(value[layer]).all():
            raise RuntimeError(f"capture {name} inactive layer {layer} sentinel changed")


def validate_capture_archive(
    path: Path,
    row: Mapping[str, Any],
    *,
    input_hashes: Mapping[str, str],
    injection_layers: Sequence[int],
    router_layers: Sequence[int],
    model_layers: int,
    hidden_size: int,
    num_experts: int,
) -> dict[str, Any]:
    """Validate a complete archive before it can be admitted by resume."""

    with np.load(path, allow_pickle=False) as source:
        expected_fields = _expected_array_fields(injection_layers)
        if len(source.files) != len(expected_fields) or set(source.files) != expected_fields:
            raise RuntimeError("capture archive member schema changed")
        expected_scalars = {
            "schema": SCHEMA,
            "request_id": str(row["request_id"]),
            "domain": str(row["domain"]),
            "split": str(row["split"]),
            "prompt_sha256": str(row["prompt_sha256"]),
            "request_manifest_row_sha256": canonical_sha256(dict(row)),
            "config_sha256": str(input_hashes["config_sha256"]),
            "manifest_sha256": str(input_hashes["manifest_sha256"]),
            "manifest_facts_sha256": str(input_hashes["manifest_facts_sha256"]),
        }
        for name, expected in expected_scalars.items():
            if str(_scalar(source, name)) != expected:
                raise RuntimeError(f"capture scalar identity changed: {name}")
        position = int(row["decode_position"])
        sequence = position + 1
        if int(_scalar(source, "decode_position")) != position:
            raise RuntimeError("capture decode position changed")
        if int(_scalar(source, "next_token_id")) != int(row["next_token_id"]):
            raise RuntimeError("capture next-token label changed")
        expected_tokens = np.asarray(row["prompt_token_ids"][:sequence], np.int64)
        input_ids = np.asarray(source["input_ids"])
        attention_mask = np.asarray(source["attention_mask"])
        if (
            input_ids.shape != (sequence,)
            or input_ids.dtype != np.int64
            or not np.array_equal(input_ids, expected_tokens)
        ):
            raise RuntimeError("capture input token prefix changed")
        if (
            attention_mask.shape != (sequence,) or attention_mask.dtype != np.int64
            or not np.array_equal(attention_mask, np.ones(sequence, np.int64))
        ):
            raise RuntimeError("capture attention mask changed")
        hidden = np.asarray(source["hidden"])
        router_logits = np.asarray(source["router_logits"])
        router_scores = np.asarray(source["router_scores"])
        router_ids = np.asarray(source["router_ids"])
        if hidden.shape != (model_layers, sequence, hidden_size) or hidden.dtype != np.float32:
            raise RuntimeError("capture hidden array shape/dtype changed")
        if (
            router_logits.shape != (model_layers, sequence, num_experts)
            or router_logits.dtype != np.float32
        ):
            raise RuntimeError("capture router-logit array shape/dtype changed")
        if (
            router_scores.shape != (model_layers, sequence, TOP_K)
            or router_scores.dtype != np.float32
        ):
            raise RuntimeError("capture router-score array shape/dtype changed")
        if router_ids.shape != (model_layers, sequence, TOP_K) or router_ids.dtype != np.int64:
            raise RuntimeError("capture router-ID array shape/dtype changed")
        active_hidden = set(map(int, injection_layers))
        active_router = set(map(int, router_layers))
        _validate_sparse_float_layers(hidden, active_layers=active_hidden, name="hidden")
        _validate_sparse_float_layers(
            router_logits, active_layers=active_router, name="router_logits",
        )
        _validate_sparse_float_layers(
            router_scores, active_layers=active_router, name="router_scores",
        )
        for layer in range(model_layers):
            if layer in active_router:
                if np.any(router_ids[layer] < 0) or np.any(router_ids[layer] >= num_experts):
                    raise RuntimeError(f"capture router IDs are invalid at layer {layer}")
            elif not np.all(router_ids[layer] == -1):
                raise RuntimeError(f"capture router-ID sentinel changed at layer {layer}")
        for layer in injection_layers:
            for prefix in ("residual", "x", "routed"):
                name = f"{prefix}_layer_{int(layer):02d}"
                value = np.asarray(source[name])
                if value.shape != (sequence, hidden_size) or value.dtype != np.float32:
                    raise RuntimeError(f"capture {name} shape/dtype changed")
                if not np.isfinite(value).all():
                    raise RuntimeError(f"capture {name} is not finite")
        repeat_json = str(_scalar(source, "current_token_repeat_facts_json"))
        repeat_digest = str(_scalar(source, "current_token_repeat_facts_sha256"))
        if hashlib.sha256(repeat_json.encode()).hexdigest() != repeat_digest:
            raise RuntimeError("current-token repeat fact digest changed")
        repeat = _validate_repeat_facts(
            _json_no_duplicate_keys(repeat_json, source=f"{path}:repeat"),
        )
        if _scalar(source, "current_token_repeat_all_exact") is not True:
            raise RuntimeError("capture current-token repeat scalar is not exact")
    return repeat


def _validate_existing(
    path: Path,
    row: Mapping[str, Any],
    *,
    trusted_record: Mapping[str, Any] | None,
    input_hashes: Mapping[str, str],
    injection_layers: Sequence[int],
    router_layers: Sequence[int],
    model_layers: int,
    hidden_size: int,
    num_experts: int,
) -> dict[str, Any] | None:
    if trusted_record is None:
        return None
    if not path.is_file():
        raise RuntimeError(f"resume facts name a missing capture: {path.name}")
    if (
        int(trusted_record.get("bytes", -1)) != path.stat().st_size
        or trusted_record.get("sha256") != sha256(path)
        or trusted_record.get("request_manifest_row_sha256") != canonical_sha256(dict(row))
    ):
        raise RuntimeError(f"resume capture hash/row identity changed: {path.name}")
    repeat = validate_capture_archive(
        path, row, input_hashes=input_hashes,
        injection_layers=injection_layers, router_layers=router_layers,
        model_layers=model_layers, hidden_size=hidden_size,
        num_experts=num_experts,
    )
    if trusted_record.get("repeat_gate_sha256") != canonical_sha256(repeat):
        raise RuntimeError(f"resume repeat gate identity changed: {path.name}")
    return repeat


def _model_dimensions(model: Any) -> tuple[int, int, int]:
    language_model = model.model.language_model
    layers = language_model.layers
    model_layers = len(layers)
    hidden_size = int(layers[0].post_attention_layernorm.weight.numel())
    candidates = [
        getattr(model, "config", None),
        getattr(language_model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ]
    num_experts: int | None = None
    for candidate in candidates:
        value = getattr(candidate, "num_experts", None)
        if value is not None:
            num_experts = int(value)
            break
    if num_experts is None:
        gate = layers[0].mlp.gate
        linear_widths = [
            int(module.out_features)
            for module in gate.modules()
            if isinstance(module, torch.nn.Linear) and int(module.out_features) > TOP_K
        ]
        if linear_widths:
            num_experts = max(linear_widths)
    if model_layers <= 0 or hidden_size <= 0 or num_experts is None or num_experts <= TOP_K:
        raise RuntimeError("could not establish full-model capture dimensions")
    return model_layers, hidden_size, num_experts


def _load_resume_records(
    *,
    facts_path: Path,
    split: str,
    input_hashes: Mapping[str, str],
    checkpoint_hashes: Mapping[str, str],
) -> dict[str, Mapping[str, Any]]:
    if not facts_path.is_file():
        return {}
    facts = load_json(facts_path)
    expected = {
        "config_sha256": input_hashes["config_sha256"],
        "manifest_sha256": input_hashes["manifest_sha256"],
        "manifest_facts_sha256": input_hashes["manifest_facts_sha256"],
        "checkpoint_config_sha256": checkpoint_hashes["checkpoint_config_sha256"],
        "checkpoint_index_sha256": checkpoint_hashes["checkpoint_index_sha256"],
    }
    if (
        facts.get("schema") != FACTS_SCHEMA
        or facts.get("completed") is not True
        or facts.get("split") != split
        or facts.get("resume_authentication")
        != "file_sha256_plus_row_and_input_pins_plus_full_array_and_repeat_gate_validation"
    ):
        raise RuntimeError("existing capture facts cannot authenticate resume")
    for name, value in expected.items():
        if facts.get(name) != value:
            raise RuntimeError(f"existing capture facts input pin changed: {name}")
    files = facts.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("existing capture facts file inventory is absent")
    records: dict[str, Mapping[str, Any]] = {}
    for name, record in files.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(record, Mapping):
            raise RuntimeError("existing capture facts contain an unsafe file record")
        if name in records:
            raise RuntimeError("existing capture facts repeat a file record")
        records[name] = record
    return records


def _gpu_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for same-host capture")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    required = config["hardware_execution_path"]
    if str(required["required_gpu_name_substring"]) not in name:
        raise RuntimeError(f"GPU gate failed: {name}")
    if memory + 1e-6 < float(required["minimum_gpu_memory_gib"]):
        raise RuntimeError(f"GPU memory gate failed: {memory:.3f} GiB")
    return {"gpu": name, "gpu_memory_gib": memory}


def capture(
    *,
    config_path: Path,
    manifest_path: Path,
    checkpoint: Path,
    output_dir: Path,
    split: str,
    request_limit: int | None,
    resume: bool,
    manifest_facts_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, rows, input_hashes = validate_capture_inputs(
        config_path=config_path, manifest_path=manifest_path,
        manifest_facts_path=manifest_facts_path,
    )
    checkpoint_hashes = validate_checkpoint_pins(config, checkpoint)
    if split != "all":
        rows = [row for row in rows if str(row["split"]) == split]
    if request_limit is not None:
        rows = rows[: int(request_limit)]
    if not rows:
        raise RuntimeError("capture selection is empty")
    hardware = _gpu_gate(config)
    layers = tuple(map(int, config["injection_layers"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model, tokenizer, load_seconds = _load_model(checkpoint)
    del tokenizer
    observer = IndependentDecodeCaptureObserver(model, layers)
    model_layers, hidden_size, num_experts = _model_dimensions(model)
    if observer.model_layers != model_layers:
        raise RuntimeError("capture observer/model layer count changed")
    facts_path = output_dir / f"capture_facts_{split}.json"
    resume_records = (
        _load_resume_records(
            facts_path=facts_path, split=split, input_hashes=input_hashes,
            checkpoint_hashes=checkpoint_hashes,
        )
        if resume else {}
    )
    files: dict[str, Any] = {}
    repeat_rows: list[dict[str, Any]] = []
    inference_seconds = 0.0
    try:
        for offset, row in enumerate(rows, start=1):
            path = capture_path(output_dir, str(row["request_id"]))
            resumed_repeat = _validate_existing(
                path, row, trusted_record=resume_records.get(path.name),
                input_hashes=input_hashes, injection_layers=layers,
                router_layers=observer.router_layers, model_layers=model_layers,
                hidden_size=hidden_size, num_experts=num_experts,
            ) if resume else None
            if resumed_repeat is not None:
                files[path.name] = {
                    "sha256": sha256(path), "bytes": path.stat().st_size,
                    "resumed": True,
                    "request_manifest_row_sha256": canonical_sha256(dict(row)),
                    "repeat_gate_sha256": canonical_sha256(resumed_repeat),
                }
                repeat_rows.append({
                    "request_id": str(row["request_id"]), "resumed": True,
                    **resumed_repeat,
                })
                print(f"[capture {offset}/{len(rows)}] reused {row['request_id']}", flush=True)
                continue
            tokens = list(map(int, row["prompt_token_ids"]))
            position = int(row["decode_position"])
            if position < 1 or position + 1 >= len(tokens):
                raise RuntimeError("manifest decode position is invalid")
            observations, prefix, prefix_seconds = _advance_exact_prefix(
                observer, model, tokens, position, device,
            )
            inference_seconds += prefix_seconds
            before_current = prefix
            first = observer.run(
                model, tokens[position], position,
                clone_decode_cache(before_current), device,
            )
            second = observer.run(
                model, tokens[position], position,
                clone_decode_cache(before_current), device,
            )
            inference_seconds += first.elapsed_seconds + second.elapsed_seconds
            repeat = _repeat_exact(first, second)
            if not repeat["all_exact"]:
                raise RuntimeError(
                    f"same-process current-token baseline drift: {row['request_id']}"
                )
            observations.append(first)
            arrays = _capture_arrays(
                row, observations, layers, observer.router_layers,
                model_layers=model_layers, input_hashes=input_hashes,
                repeat_facts=repeat,
            )
            atomic_npz(path, **arrays)
            files[path.name] = {
                "sha256": sha256(path), "bytes": path.stat().st_size,
                "resumed": False,
                "request_manifest_row_sha256": canonical_sha256(dict(row)),
                "repeat_gate_sha256": canonical_sha256(repeat),
            }
            repeat_rows.append({
                "request_id": str(row["request_id"]), "resumed": False,
                **repeat,
            })
            print(
                f"[capture {offset}/{len(rows)}] {row['request_id']} "
                f"position={position} bytes={path.stat().st_size}", flush=True,
            )
            del prefix, before_current, first, second, observations, arrays, repeat
            gc.collect()
    finally:
        observer.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    facts = {
        "completed": True,
        "schema": FACTS_SCHEMA,
        "resume_authentication": (
            "file_sha256_plus_row_and_input_pins_plus_full_array_and_repeat_gate_validation"
        ),
        "config": str(config_path),
        "config_sha256": input_hashes["config_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": input_hashes["manifest_sha256"],
        "manifest_facts": input_hashes["manifest_facts_path"],
        "manifest_facts_sha256": input_hashes["manifest_facts_sha256"],
        "input_identity_is_byte_pinned_not_path_pinned": True,
        "checkpoint": str(checkpoint),
        **checkpoint_hashes,
        "split": split,
        "requests": len(rows),
        "request_ids": [str(row["request_id"]) for row in rows],
        "freshly_captured_requests": sum(not bool(row["resumed"]) for row in repeat_rows),
        "resumed_requests": sum(bool(row["resumed"]) for row in repeat_rows),
        "freshly_captured_request_ids": [
            row["request_id"] for row in repeat_rows if not bool(row["resumed"])
        ],
        "resumed_request_ids": [
            row["request_id"] for row in repeat_rows if bool(row["resumed"])
        ],
        "layers": list(layers),
        "array_model_layers": model_layers,
        "array_hidden_size": hidden_size,
        "array_routed_experts": num_experts,
        "files": files,
        "repeat_gates": repeat_rows,
        "all_current_token_repeats_exact": (
            len(repeat_rows) == len(rows)
            and all(bool(row["all_exact"]) for row in repeat_rows)
        ),
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - started,
        "exact_prefix_progression": True,
        "one_isolated_decode_token_per_request": True,
        "terminal_logits_retained_or_written": False,
        "candidate_state_or_delta_used": False,
        "downstream_candidate_outcome_used": False,
        "environment": {
            **hardware,
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    atomic_json(facts_path, facts)
    return facts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-facts", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("calibration", "evaluation", "all"), default="all",
    )
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.request_limit is not None and args.request_limit < 1:
        raise ValueError("request-limit must be positive")
    facts = capture(
        config_path=args.config,
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        request_limit=args.request_limit,
        resume=bool(args.resume),
        manifest_facts_path=args.manifest_facts,
    )
    print(json.dumps(facts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
