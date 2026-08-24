#!/usr/bin/env python3
"""Run parity gates and matched-rate D1 tests on compact layer slices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_layer_slice import (  # noqa: E402
    load_d1_layer_slice,
    slice_parity,
)


SCHEMA = "pr13_d1_layer_slice_pilot_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("parity",), default="parity")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 12, 23])
    parser.add_argument(
        "--requests", nargs="+",
        default=["mxfp4-confirm-006", "mxfp4-confirm-007", "mxfp4-confirm-008"],
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _capture_path(directory: Path, request_id: str) -> Path:
    candidates = (
        directory / f"{request_id}.npz",
        directory / "baseline" / f"{request_id}.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"capture not found for {request_id}: {directory}")


def run_parity(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for exact slice parity")
    rows = []
    input_hashes: dict[str, str] = {}
    layer_load_seconds: dict[str, float] = {}
    for layer in map(int, args.layers):
        started = time.perf_counter()
        model_slice = load_d1_layer_slice(
            args.checkpoint, layer, device=args.device,
        )
        torch.cuda.synchronize()
        layer_load_seconds[str(layer)] = time.perf_counter() - started
        for request_id in map(str, args.requests):
            capture_path = _capture_path(args.capture_dir, request_id)
            input_hashes[str(capture_path)] = sha256(capture_path)
            with np.load(capture_path, allow_pickle=False) as capture:
                residual = np.asarray(
                    capture[f"residual_layer_{layer:02d}"], np.float32,
                )
                x = np.asarray(capture[f"x_layer_{layer:02d}"], np.float32)
                routed = np.asarray(
                    capture[f"routed_layer_{layer:02d}"], np.float32,
                )
                hidden_reference = np.asarray(
                    capture["hidden"][layer], np.float32,
                )
                router_reference = np.asarray(
                    capture["router_logits"][layer + 1], np.float32,
                )
                attention_mask = (
                    np.asarray(capture["attention_mask"])
                    if "attention_mask" in capture.files
                    else np.ones(len(capture["input_ids"]), np.int64)
                )
            torch.cuda.synchronize()
            executed = time.perf_counter()
            with torch.inference_mode():
                hidden = model_slice.compose_current_output(
                    residual, x, routed,
                )
                router = model_slice.next_router_logits(hidden, attention_mask)
                hidden_repeat = model_slice.compose_current_output(
                    residual, x, routed,
                )
                router_repeat = model_slice.next_router_logits(
                    hidden_repeat, attention_mask,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - executed
            parity = slice_parity(
                hidden, hidden_reference, router, router_reference,
            )
            row = {
                "schema": SCHEMA,
                "phase": "parity",
                "layer": layer,
                "next_layer": layer + 1,
                "request_id": request_id,
                "sequence_length": int(hidden_reference.shape[0]),
                "slice_runtime_ms": 1000.0 * elapsed,
                "next_mixer_type": str(model_slice.next_mixer_type),
                "paired_hidden_bit_identical": bool(
                    torch.equal(hidden, hidden_repeat)
                ),
                "paired_router_logits_bit_identical": bool(
                    torch.equal(router, router_repeat)
                ),
                **parity.__dict__,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del model_slice
        torch.cuda.empty_cache()
    hard_gate = all(
        row["paired_hidden_bit_identical"]
        and row["paired_router_logits_bit_identical"]
        for row in rows
    )
    return {
        "schema": SCHEMA,
        "phase": "parity",
        "hard_gate_passed": hard_gate,
        "hard_gate_definition": (
            "same-process repeated current hidden and next-router logits are "
            "bit-identical for every request/layer; stored RTX PRO 6000 "
            "captures are reported as cross-device drift diagnostics only"
        ),
        "layers": list(map(int, args.layers)),
        "requests": list(map(str, args.requests)),
        "rows": rows,
        "layer_load_seconds": layer_load_seconds,
        "input_sha256": input_hashes,
        "checkpoint_index_sha256": sha256(
            args.checkpoint / "model.safetensors.index.json",
        ),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slice_core_sha256": sha256(
            EXPERIMENT / "src/oracle_study/d1_layer_slice.py",
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_parity(args)
    output = args.output_dir / "parity" / "d1_slice_parity.json"
    atomic_json(output, result)
    print(f"wrote {output}", flush=True)
    if not result["hard_gate_passed"]:
        raise RuntimeError("D1 slice parity hard gate failed")


if __name__ == "__main__":
    main()
