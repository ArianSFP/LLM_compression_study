#!/usr/bin/env python3
"""Audit 3090 allocation routes against the exact PRO cached-decode path."""

from __future__ import annotations

import argparse
import gc
import json
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

from oracle_study.d1_decode import clone_decode_cache  # noqa: E402
from oracle_study.d1_decode_tail import (  # noqa: E402
    sha256,
    validate_cached_decode_tail_config,
)
from run_d1_downstream_tail_kl import (  # noqa: E402
    CachedDecodeObserver,
    _gpu_gate,
    _policy_banks,
)
from run_same_host_causal_controls import (  # noqa: E402
    _encoded_requests,
    _load_model,
    atomic_json,
    atomic_parquet,
)


SCHEMA = "pr13_d1_tail_route_transfer_audit_v1"
TABLE = "d1_tail_route_transfer.parquet"
FACTS = "d1_tail_route_transfer_facts.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def _margin(logits: np.ndarray, selected_ids: np.ndarray) -> float:
    values = np.asarray(logits, np.float64).reshape(-1)
    selected = np.asarray(selected_ids, np.int64).reshape(-1)
    selected_set = set(selected.tolist())
    outsiders = np.asarray(
        [index for index in range(values.size) if index not in selected_set],
        np.int64,
    )
    if selected.size != 8 or outsiders.size != values.size - 8:
        raise ValueError("route margin requires eight unique selected experts")
    return float(np.min(values[selected]) - np.max(values[outsiders]))


def audit(
    config: Mapping[str, Any],
    config_path: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    hardware = _gpu_gate(config)
    _, fixed, _, banks = _policy_banks(config)
    rates = list(map(int, config["page_caps"]))
    layers = list(map(int, config["injection_layers"]))
    reference_banks = {}
    for layer in layers:
        reference = banks[(layer, rates[0])]
        for rate in rates[1:]:
            candidate = banks[(layer, rate)]
            if not np.array_equal(reference.expert_ids, candidate.expert_ids):
                raise RuntimeError(
                    f"source expert identities vary by rate at layer {layer}"
                )
        reference_banks[layer] = reference

    model, tokenizer, load_seconds = _load_model(checkpoint)
    encoded = _encoded_requests(tokenizer, model, config)
    observer = CachedDecodeObserver(model, layers[0])
    rows: list[dict[str, Any]] = []
    baseline_seconds = 0.0
    try:
        identity = reference_banks[layers[0]].identity
        group_lookup = {
            (str(row.request_id), int(row.position)): int(row.group)
            for row in identity.itertuples(index=False)
        }
        for request_id in map(str, config["validation_request_ids"]):
            request = encoded[request_id]
            first = observer.run(model, request, 0, None)
            baseline_seconds += first.elapsed_seconds
            exact_prefix = first.cache
            del first
            positions = int(
                config["position_mask"]["admitted_positions_per_request"][request_id]
            )
            for position in range(1, positions + 1):
                observation = observer.run(
                    model,
                    request,
                    position,
                    clone_decode_cache(exact_prefix),
                )
                baseline_seconds += observation.elapsed_seconds
                group = group_lookup[(request_id, position)]
                for layer in layers:
                    bank = reference_banks[layer]
                    source_ids = np.asarray(bank.expert_ids[group], np.int64)
                    live_ids = (
                        observation.router_ids[layer]
                        .reshape(-1)
                        .numpy()
                        .astype(np.int64)
                    )
                    logits = (
                        observation.router_logits[layer]
                        .reshape(-1)
                        .float()
                        .numpy()
                        .astype(np.float64)
                    )
                    source_set = set(source_ids.tolist())
                    live_set = set(live_ids.tolist())
                    rows.append({
                        "schema": SCHEMA,
                        "layer": layer,
                        "next_layer": layer + 1,
                        "group": group,
                        "request_id": request_id,
                        "position": position,
                        "source_ids": json.dumps(source_ids.tolist(), separators=(",", ":")),
                        "live_ids": json.dumps(live_ids.tolist(), separators=(",", ":")),
                        "route_set_equal": source_set == live_set,
                        "route_order_equal": bool(np.array_equal(source_ids, live_ids)),
                        "membership_pairs_changed": len(source_set - live_set),
                        "source_labeled_margin_on_live_logits": _margin(logits, source_ids),
                        "live_rank8_rank9_margin": _margin(logits, live_ids),
                        "entered_experts": json.dumps(
                            sorted(live_set - source_set), separators=(",", ":"),
                        ),
                        "left_experts": json.dumps(
                            sorted(source_set - live_set), separators=(",", ":"),
                        ),
                    })
                exact_prefix = observation.cache
                del observation
                gc.collect()
    finally:
        observer.close()
        del encoded, tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).sort_values(
        ["layer", "group"], kind="stable",
    ).reset_index(drop=True)
    expected = len(layers) * int(config["position_mask"]["expected_groups_per_layer"])
    if len(frame) != expected:
        raise RuntimeError(f"route audit grid is incomplete: {len(frame)} != {expected}")
    output.mkdir(parents=True, exist_ok=True)
    table = output / TABLE
    atomic_parquet(table, frame)
    mismatches = frame[~frame["route_set_equal"]]
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "rows": len(frame),
        "route_set_mismatches": len(mismatches),
        "route_set_mismatch_rate": float(len(mismatches) / len(frame)),
        "route_order_mismatches": int((~frame["route_order_equal"]).sum()),
        "layers_with_set_mismatch": sorted(
            mismatches["layer"].astype(int).unique().tolist()
        ),
        "minimum_source_labeled_margin": float(
            frame["source_labeled_margin_on_live_logits"].min()
        ),
        "table": {
            "path": str(table),
            "sha256": sha256(table),
            "bytes": table.stat().st_size,
        },
        "fixed_d1_source_policy_by_rate": {
            str(rate): policy for rate, policy in sorted(fixed.items())
        },
        "model_load_seconds": load_seconds,
        "baseline_seconds": baseline_seconds,
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            **hardware,
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "allocation_transfer_admitted": len(mismatches) == 0,
        "test_rows_admitted_or_used": False,
    }
    atomic_json(output / FACTS, facts)
    return facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    validate_cached_decode_tail_config(config)
    print(json.dumps(
        audit(config, args.config, args.checkpoint, args.output),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
