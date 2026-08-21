#!/usr/bin/env python3
"""Benchmark one matrix-free interaction factor on a locked MXFP4 expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.average_rate_allocator import (  # noqa: E402
    lagrangian_rate_frontier, randomized_joint_metric_factor,
)
from oracle_study.interaction_field import (  # noqa: E402
    encode_interaction_factor, make_encoded_factor_self_safe,
)
from oracle_study.neuron_selector import unit_score_metadata  # noqa: E402
from oracle_study.split_interaction_field import (  # noqa: E402
    build_split_interaction_field, split_projection_responses,
)
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_set_utility_distillation as study  # noqa: E402
import run_sparse_streaming_study as base  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=62)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    data = study.load_capture_admitted(args.captures, ["train"])
    local = study.layer_view(data, args.layer)
    proxy, proxy_facts = base.proxy_gradients(
        local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)],
        local["split"],
    )
    index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(records[name]) for name in study.PROJECTIONS}
    started = time.perf_counter()
    decoded = study.decode_expert(
        args.checkpoint, index, trees, args.layer, args.expert,
    )
    decode_seconds = time.perf_counter() - started
    q2 = decoded["down"][0]
    q4 = decoded["down"][2]
    started = time.perf_counter()
    factor = randomized_joint_metric_factor(
        q2, q4, proxy, float(proxy_facts["beta"]), args.rank,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        seed=int(config.get("seed", 20260821)) + 1000 * args.layer + args.expert,
        device=args.device,
    )
    fit_seconds = time.perf_counter() - started
    abc = unit_score_metadata(q2, q4, proxy=proxy, beta=float(proxy_facts["beta"]))
    encoded = encode_interaction_factor(factor, "int4_per_row", hadamard_rotate=True)
    encoded, _ = make_encoded_factor_self_safe(encoded, abc)
    decoded_factor = encoded.decode()
    activation = np.asarray(local["x"][0], np.float32)
    q2_all = tuple(decoded[name][0] for name in study.PROJECTIONS)
    q4_all = tuple(decoded[name][2] for name in study.PROJECTIONS)
    responses = split_projection_responses(q2_all, q4_all, activation)
    field = build_split_interaction_field(decoded_factor, responses.hidden, abc)
    frontier_started = time.perf_counter()
    trace = lagrangian_rate_frontier(
        field,
        maximum_pages=int(config.get("frontier_maximum_pages", 1536)),
        target_pages=config.get(
            "frontier_target_pages", [0, 384, 576, 749, 768, 1152, 1536],
        ),
        price_ratios=config.get(
            "frontier_price_ratios", [16, 4, 1, .25, 0],
        ),
        coordinate_sweeps=int(config.get("coordinate_sweeps", 8)),
        local_shortlist=int(config.get("local_shortlist", 8)),
        local_swap_units=int(config.get("local_swap_units", 3)),
        local_max_passes=int(config.get("local_max_passes", 6)),
        price_local_max_passes=int(config.get("price_local_max_passes", 0)),
    )
    frontier_seconds = time.perf_counter() - frontier_started

    rng = np.random.default_rng(91)
    probes = rng.normal(size=(2 * q2.shape[1], 32)).astype(np.float32)
    joint = np.concatenate((q4, q2), axis=1).astype(np.float64)
    exact_output = joint @ probes
    exact = np.sum(exact_output * exact_output, axis=0)
    projected = exact_output.T @ np.asarray(proxy, np.float64)
    exact += float(proxy_facts["beta"]) * np.sum(projected * projected, axis=1)
    rows = np.concatenate((factor.l4, factor.l2), axis=0).astype(np.float64)
    approximate = np.sum((rows.T @ probes) ** 2, axis=0)
    relative_probe_error = np.abs(approximate - exact) / np.maximum(exact, 1e-30)
    print(json.dumps({
        "layer": args.layer,
        "expert": args.expert,
        "rank": factor.rank,
        "oversample": args.oversample,
        "power_iterations": args.power_iterations,
        "decode_seconds": decode_seconds,
        "factor_fit_seconds": fit_seconds,
        "frontier_seconds": frontier_seconds,
        "frontier_options": len(trace.options),
        "frontier_coordinate_sweeps": trace.coordinate_sweeps,
        "frontier_local_passes": trace.local_passes,
        "probe_relative_error_median": float(np.median(relative_probe_error)),
        "probe_relative_error_p90": float(np.quantile(relative_probe_error, .9)),
        "runtime_provenance": study.runtime_provenance(args.device),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
