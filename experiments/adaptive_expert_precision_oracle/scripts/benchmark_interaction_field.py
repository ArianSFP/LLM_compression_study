#!/usr/bin/env python3
"""One-invocation real-size benchmark for the low-rank interaction field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.interaction_field import (  # noqa: E402
    build_interaction_field,
    factor_exact_proxy_plus_tail,
    factor_joint_gram,
    field_coordinate_descent,
    field_forward_greedy,
    field_local_search,
    relaxed_field_descent,
    round_relaxed_pages,
    truncate_interaction_factor,
)
from oracle_study.neuron_selector import (  # noqa: E402
    complete_unit_scores,
    factorized_unit_outputs,
    unit_score_metadata,
)
from oracle_study.set_utility_oracles import STATE_GD, hybrid_state_output  # noqa: E402
from oracle_study.unit_set_teacher import down_metric_gram  # noqa: E402
from run_mxfp4_selective_pages import tree_from_record  # noqa: E402
import run_set_utility_distillation as prior  # noqa: E402
import run_sparse_streaming_study as base  # noqa: E402


def _timed(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def _exact_damage(outputs, states, proxy, beta):
    output = hybrid_state_output(outputs, states)
    residual = np.asarray(outputs.target_output, np.float64) - output
    return float(base.numpy_qenergy(residual, proxy=proxy, beta=beta))


def _best_coordinate(field, seeds, budget, sweeps):
    traces = {
        name: field_coordinate_descent(field, seed, budget, max_sweeps=sweeps)
        for name, seed in sorted(seeds.items())
    }
    name, trace = min(traces.items(), key=lambda item: (item[1].damage, item[0]))
    return name, trace, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr10-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int)
    parser.add_argument("--invocation-index", type=int, default=0)
    parser.add_argument("--budget-pages", type=int, default=768)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--coordinate-sweeps", type=int, default=8)
    parser.add_argument("--relax-iterations", type=int, default=128)
    parser.add_argument("--local-passes", type=int, default=12)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    pr10 = json.loads(args.pr10_config.read_text())
    if prior.sha256(args.pr10_config) != config["pr10_config_sha256"]:
        raise RuntimeError("PR #10 config hash changed")
    base.verify_locked_inputs(
        pr10, args.pr10_config, args.exact_captures, args.checkpoint, args.trees,
        "exact_checkpoint",
    )
    data = prior.load_capture_admitted(args.exact_captures, ["train", "validation"])
    plan = prior.evaluation_plan(data, pr10)
    prior.audit_validation_plan(plan, pr10)
    matches = [entry for entry in plan if int(entry[0]) == args.layer]
    if args.expert is not None:
        matches = [entry for entry in matches if int(entry[1]) == args.expert]
    if len(matches) != 1:
        raise RuntimeError(f"benchmark cell selection is ambiguous: {[(x[0], x[1]) for x in matches]}")
    layer, expert, stratum, records, ranks = matches[0]
    if args.invocation_index < 0 or args.invocation_index >= len(records):
        raise ValueError("invocation-index lies outside the selected cell")
    record = int(records[args.invocation_index])
    router_rank = int(ranks[args.invocation_index])
    local = prior.layer_view(data, layer)
    proxy, proxy_facts = base.proxy_gradients(
        local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)], local["split"],
    )
    beta = float(proxy_facts["beta"])
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: tree_from_record(tree_records[name]) for name in prior.PROJECTIONS}
    matrices, decode_seconds = _timed(
        prior.decode_expert, args.checkpoint, index, trees, layer, expert,
    )
    q2 = tuple(matrices[name][0] for name in prior.PROJECTIONS)
    q4 = tuple(matrices[name][2] for name in prior.PROJECTIONS)
    gram, gram_seconds = _timed(
        down_metric_gram, q2[2], q4[2], proxy=proxy, beta=beta, dtype=np.float64,
    )
    abc = unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
    activation = np.asarray(local["x"][record], np.float32)
    outputs = factorized_unit_outputs(q2, q4, activation)
    all_00 = np.zeros(outputs.units, np.int64)
    base_damage = _exact_damage(outputs, all_00, proxy, beta)
    independent = all_00.copy()
    coherent = min(outputs.units, args.budget_pages // 3)
    order = np.lexsort((np.arange(outputs.units), -complete_unit_scores(outputs.h2, outputs.h4, abc)))
    independent[order[:coherent]] = STATE_GD

    fitted = {}
    timings = {}
    for name, builder in (
        ("joint_eigh", lambda: factor_joint_gram(gram, args.max_rank, method="eigh")),
        (
            "joint_pivoted_cholesky",
            lambda: factor_joint_gram(gram, args.max_rank, method="pivoted_cholesky"),
        ),
        (
            "exact_proxy_plus_eigh_tail",
            lambda: factor_exact_proxy_plus_tail(
                q2[2], q4[2], proxy, beta, args.max_rank, method="eigh",
            ),
        ),
    ):
        fitted[name], timings[name] = _timed(builder)

    results = []
    ranks_to_run = [rank for rank in (4, 8, 16, 32) if rank <= args.max_rank]
    for family, full in fitted.items():
        tail_ranks = ([0] + ranks_to_run) if full.exact_rank else ranks_to_run
        for tail_rank in tail_ranks:
            factor = truncate_interaction_factor(full, tail_rank, dtype=np.float32)
            field = build_interaction_field(factor, outputs.h2, outputs.h4, abc)
            relaxation_step = 1.0 / max(
                2.0 * float(np.einsum("usr,usr->", field.rho, field.rho, optimize=True)),
                1.0,
            )
            forward, forward_seconds = _timed(
                field_forward_greedy, field, [args.budget_pages],
            )
            forward_snapshot = forward.snapshots[args.budget_pages]
            relaxed, relaxed_seconds = _timed(
                relaxed_field_descent,
                field,
                args.budget_pages,
                max_iterations=args.relax_iterations,
                tolerance=1e-8,
                step_size=relaxation_step,
            )
            relaxed_states = round_relaxed_pages(relaxed.probabilities, args.budget_pages)
            seeds = {
                "all_00": all_00,
                "independent_abc": independent,
                "residual_forward": forward_snapshot.states,
                "relaxed_round": relaxed_states,
            }
            started = time.perf_counter()
            seed_name, coordinate, coordinate_traces = _best_coordinate(
                field, seeds, args.budget_pages, args.coordinate_sweeps,
            )
            coordinate_seconds = time.perf_counter() - started
            cheap_started = time.perf_counter()
            cheap_seed_name, cheap_coordinate, cheap_traces = _best_coordinate(
                field,
                {name: seeds[name] for name in ("all_00", "independent_abc")},
                args.budget_pages,
                args.coordinate_sweeps,
            )
            cheap_coordinate_seconds = time.perf_counter() - cheap_started
            cheap_repaired, cheap_local_seconds = _timed(
                field_local_search,
                field,
                cheap_coordinate.states,
                args.budget_pages,
                shortlist_size=8,
                max_swap_units=3,
                max_passes=args.local_passes,
                page_balanced=False,
            )
            cheap_exact_damage = _exact_damage(
                outputs, cheap_repaired.states, proxy, beta,
            )
            repaired, local_seconds = _timed(
                field_local_search,
                field,
                coordinate.states,
                args.budget_pages,
                shortlist_size=8,
                max_swap_units=3,
                max_passes=args.local_passes,
                page_balanced=False,
            )
            exact_damage = _exact_damage(outputs, repaired.states, proxy, beta)
            results.append({
                "factor_family": family,
                "tail_rank": tail_rank,
                "exact_rank": factor.exact_rank,
                "total_rank": factor.rank,
                "factor_payload_bytes": factor.payload_bytes,
                "factor_fit_seconds_shared": timings[family],
                "forward_seconds": forward_seconds,
                "relaxed_seconds": relaxed_seconds,
                "coordinate_seconds": coordinate_seconds,
                "local_seconds": local_seconds,
                "cheap_coordinate_seconds": cheap_coordinate_seconds,
                "cheap_local_seconds": cheap_local_seconds,
                "cheap_selected_seed": cheap_seed_name,
                "cheap_pages": cheap_repaired.pages,
                "cheap_compressed_damage": cheap_repaired.damage,
                "cheap_exact_recovery": 1.0 - cheap_exact_damage / base_damage,
                "cheap_coordinate_interaction_dot_macs": sum(
                    trace.interaction_dot_macs for trace in cheap_traces.values()
                ),
                "selected_seed": seed_name,
                "pages": repaired.pages,
                "compressed_damage": repaired.damage,
                "exact_damage": exact_damage,
                "damage_relative_error": (repaired.damage - exact_damage) / base_damage,
                "exact_recovery": 1.0 - exact_damage / base_damage,
                "coordinate_interaction_dot_macs": sum(
                    trace.interaction_dot_macs for trace in coordinate_traces.values()
                ),
                "forward_candidate_evaluations": forward.candidate_evaluations,
                "relaxed_objective": relaxed.objective,
                "relaxed_expected_pages": relaxed.expected_pages,
            })

    identity = base.common_metadata(
        "exact_checkpoint", "validation", local, record, router_rank,
        layer, expert, stratum,
    )
    print(json.dumps({
        "identity": identity,
        "budget_pages": args.budget_pages,
        "decode_seconds": decode_seconds,
        "static_gram_seconds": gram_seconds,
        "proxy_rank": int(np.asarray(proxy).shape[1]),
        "base_damage": base_damage,
        "independent_recovery": 1.0 - _exact_damage(outputs, independent, proxy, beta) / base_damage,
        "results": results,
    }, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
