"""Read the immutable GCRP-2R event/sidecar capture without rewriting it."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DTYPES = {
    "bf16": (np.dtype("<u2"), "bf16"),
    "float16": (np.dtype("<f2"), "float"),
    "float32": (np.dtype("<f4"), "float"),
    "float64": (np.dtype("<f8"), "float"),
    "int64": (np.dtype("<i8"), "int"),
    "int32": (np.dtype("<i4"), "int"),
    "int16": (np.dtype("<i2"), "int"),
    "uint8": (np.dtype("u1"), "int"),
    "bool": (np.dtype("u1"), "int"),
}


def split_for_sequence(sequence_id: str, seed: int) -> str:
    """Stable 60/20/20 request-level split."""
    value = int(hashlib.sha256(f"{seed}|{sequence_id}".encode()).hexdigest()[:16], 16) % 10
    return "train" if value < 6 else ("validation" if value < 8 else "test")


class SidecarReader:
    def __init__(self, segment: Path) -> None:
        self.segment = segment
        self.maps: dict[Path, np.memmap] = {}

    def tensor(self, event: dict[str, Any]) -> np.ndarray:
        dtype, category = DTYPES[event["native_dtype"]]
        path = self.segment / event["payload_file"]
        mapping = self.maps.get(path)
        if mapping is None:
            mapping = np.memmap(path, mode="r", dtype=np.uint8)
            self.maps[path] = mapping
        offset = int(event["payload_offset"])
        length = int(event["payload_bytes"])
        values = np.frombuffer(memoryview(mapping[offset : offset + length]), dtype=dtype)
        shape = tuple(int(v) for v in event["shape"])
        if values.size != math.prod(shape):
            raise ValueError(f"shape mismatch in tensor event {event['event_id']}")
        values = values.reshape(shape)
        if category == "bf16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(values).copy()

    def close(self) -> None:
        self.maps.clear()


CURRENT_ROLES = {
    "normalized_target_router_input_a": "x",
    "post_moe_residual_xplus": "xplus",
    "routed_expert_output_delta_r": "routed_bf16",
    "raw_target_router_logits": "router_logits",
    "selected_expert_ids": "expert_ids",
    "selected_execution_weights": "router_weights",
}
FUTURE_ROLES = {
    "post_moe_residual_xplus": "xplus",
    "raw_target_router_logits": "router_logits",
    "selected_expert_ids": "expert_ids",
}


def _segment_dirs(capture_root: Path) -> list[Path]:
    result = sorted((capture_root / "segments").glob("gcrp2r_tf_bf16_seg_*"))
    return [p for p in result if (p / "CAPTURE_AUDIT.json").is_file()]


def extract_records(
    capture_root: Path,
    layers: Iterable[int],
    stride: int,
    seed: int,
    progress_every: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract current positions and H1--H4 aligned reference states.

    Every selected current position has ``absolute_position % stride == 0``.
    The following four positions are retained only for reference hidden/router
    values. All tensors are copied from immutable sidecars.
    """
    layers = {int(v) for v in layers}
    current: dict[tuple[str, int, int], dict[str, Any]] = {}
    future: dict[tuple[str, int, int], dict[str, Any]] = {}
    request_ids: dict[str, str] = {}
    manifests: list[dict[str, Any]] = []
    segments = _segment_dirs(capture_root)
    if not segments:
        raise FileNotFoundError(f"no audited segments under {capture_root}")

    for segment_index, segment in enumerate(segments):
        audit = json.loads((segment / "CAPTURE_AUDIT.json").read_text())
        if not audit.get("passed"):
            raise ValueError(f"capture audit did not pass: {segment}")
        manifest = json.loads((segment / "run_manifest.json").read_text())
        manifests.append(manifest)
        reader = SidecarReader(segment)
        wanted: dict[int, tuple[dict[str, Any], dict[str, str]]] = {}
        with (segment / "events.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                kind = event.get("event")
                if kind == "sequence_start":
                    request_ids[str(event["sequence_id"])] = str(event["request_id"])
                elif kind == "target_layer":
                    layer = int(event["target_layer"])
                    if layer not in layers:
                        continue
                    pos = int(event["absolute_sequence_position"])
                    remainder = pos % stride
                    if remainder > 4:
                        continue
                    seq = str(event["sequence_id"])
                    key = (seq, pos, layer)
                    base = {
                        "sequence_id": seq,
                        "request_id": str(event["request_id"]),
                        "position": pos,
                        "layer": layer,
                        "block_type": str(event["block_type"]),
                        "split": split_for_sequence(seq, seed),
                        "is_prompt": False,
                        "prefix_hash": str(event["authoritative_prefix_hash"]),
                    }
                    roles = CURRENT_ROLES if remainder == 0 else FUTURE_ROLES
                    target = current if remainder == 0 else future
                    target[key] = base
                    wanted[int(event["event_id"])] = (base, roles)
                elif kind == "tensor":
                    parent = int(event["parent_event_id"])
                    found = wanted.get(parent)
                    if found is None:
                        continue
                    base, roles = found
                    role = event.get("tensor_role")
                    if role in roles:
                        base[roles[role]] = reader.tensor(event)
        reader.close()
        if (segment_index + 1) % progress_every == 0 or segment_index + 1 == len(segments):
            print(json.dumps({
                "extract_segments_done": segment_index + 1,
                "extract_segments_total": len(segments),
                "current_rows": len(current),
                "future_rows": len(future),
            }), flush=True)

    required_current = set(CURRENT_ROLES.values())
    required_future = set(FUTURE_ROLES.values())
    bad_current = [k for k, v in current.items() if not required_current.issubset(v)]
    bad_future = [k for k, v in future.items() if not required_future.issubset(v)]
    if bad_current or bad_future:
        raise ValueError(f"incomplete records: current={len(bad_current)} future={len(bad_future)}")

    rows: list[dict[str, Any]] = []
    for key, record in sorted(current.items()):
        seq, pos, layer = key
        if any((seq, pos + h, layer) not in future for h in range(1, 5)):
            continue
        row = dict(record)
        for horizon in range(1, 5):
            ref = future[(seq, pos + horizon, layer)]
            row[f"h{horizon}_xplus"] = ref["xplus"]
            row[f"h{horizon}_router_logits"] = ref["router_logits"]
            row[f"h{horizon}_expert_ids"] = ref["expert_ids"]
        rows.append(row)
    if not rows:
        raise ValueError("no complete H1--H4 rows")

    scalar_names = [
        "sequence_id", "request_id", "position", "layer", "block_type",
        "split", "prefix_hash",
    ]
    array_names = [
        "x", "xplus", "routed_bf16", "router_logits", "expert_ids", "router_weights",
    ] + [f"h{h}_{name}" for h in range(1, 5) for name in ("xplus", "router_logits", "expert_ids")]
    arrays: dict[str, np.ndarray] = {}
    for name in scalar_names:
        arrays[name] = np.asarray([row[name] for row in rows])
    for name in array_names:
        arrays[name] = np.stack([row[name] for row in rows])

    split_counts = {name: int(np.sum(arrays["split"] == name)) for name in ("train", "validation", "test")}
    layer_counts = {str(layer): int(np.sum(arrays["layer"] == layer)) for layer in sorted(layers)}
    unique_sequences = {name: int(len(set(arrays["sequence_id"][arrays["split"] == name].tolist()))) for name in split_counts}
    capture_facts = {
        "segments": len(segments),
        "rows": len(rows),
        "split_rows": split_counts,
        "split_sequences": unique_sequences,
        "layer_rows": layer_counts,
        "position_stride": stride,
        "source_run_ids": [m["run_id"] for m in manifests],
        "model_revision": manifests[0]["model_revision"],
        "model_config_hash": manifests[0]["model_config_hash"],
        "tokenizer_hash": manifests[0]["tokenizer_hash"],
        "source_prompt_manifest_sha256": manifests[0]["source_prompt_manifest_sha256"],
        "capture_schema": manifests[0]["schema"],
        "quantization_format": manifests[0]["quantization_format"],
    }
    return arrays, capture_facts


def save_extraction(path: Path, arrays: dict[str, np.ndarray], facts: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".manifest.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
