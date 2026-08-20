#!/usr/bin/env python3
"""Benchmark the exact hybrid unit oracle on one locked validation invocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_set_utility_distillation as study
from oracle_study.neuron_selector import factorized_unit_outputs
from oracle_study.set_utility_oracles import exact_hybrid_unit_base_gram_greedy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--expert", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--full-hybrid", action="store_true",
        help="also time shared factorized paths and every bounded local-search basin",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    study.validate_study_contract(config)
    study.base.verify_locked_inputs(
        config,
        args.config,
        args.exact_captures,
        args.checkpoint,
        args.trees,
        "exact_checkpoint",
    )
    captured = study.load_capture_admitted(args.exact_captures, ["train", "validation"])
    local = study.layer_view(captured, args.layer)
    rows, ranks = study.occurrence(local, args.expert, "validation", 1)
    if len(rows) != 1:
        raise RuntimeError("requested expert has no locked validation occurrence")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {
        projection: study.tree_from_record(tree_records[projection])
        for projection in study.PROJECTIONS
    }
    matrices, _ = study.load_fit_expert(
        args.checkpoint, index, trees, args.layer, args.expert,
    )
    row = int(rows[0])
    activation = np.asarray(local["x"][row], np.float32)
    proxy, proxy_facts = study.base.proxy_gradients(
        local["xplus"],
        [local[f"h{h}_router_logits"] for h in range(1, 5)],
        local["split"],
    )
    q2 = tuple(matrices[name][0] for name in study.PROJECTIONS)
    q4 = tuple(matrices[name][2] for name in study.PROJECTIONS)
    outputs = factorized_unit_outputs(q2, q4, activation)
    trace = exact_hybrid_unit_base_gram_greedy(
        outputs,
        page_budgets=(384, 576, 768),
        proxy=proxy,
        beta=float(proxy_facts["beta"]),
        device=torch.device(args.device),
    )
    direct_base = study.base.numpy_qenergy(
        np.asarray(outputs.target_output) - np.asarray(outputs.base_output),
        proxy=proxy,
        beta=float(proxy_facts["beta"]),
    )
    if not np.isfinite(direct_base) or direct_base <= 0.0:
        raise RuntimeError("nonpositive or nonfinite direct baseline damage")
    snapshots = {}
    for budget, snapshot in sorted(trace.snapshots.items()):
        direct_damage = study.base.numpy_qenergy(
            np.asarray(outputs.target_output) - np.asarray(snapshot.output),
            proxy=proxy,
            beta=float(proxy_facts["beta"]),
        )
        if not np.isclose(direct_damage, snapshot.residual_energy, rtol=3e-4, atol=1e-4):
            raise RuntimeError(f"Gram/direct qenergy mismatch at page budget {budget}")
        snapshots[str(budget)] = {
            "pages": int(snapshot.pages),
            "actions": int(len(snapshot.order)),
            "recovery": float(1.0 - direct_damage / direct_base),
            "residual_energy": float(direct_damage),
        }
    full_hybrid = None
    if args.full_hybrid:
        static_down_metric = study.down_metric_gram(
            q2[2], q4[2], proxy=proxy, beta=float(proxy_facts["beta"]),
        )
        gram = study.unit_correction_gram(
            outputs.h2, outputs.h4, static_down_metric,
        )
        exact_abc = study.unit_score_metadata(
            q2[2], q4[2], proxy=proxy, beta=float(proxy_facts["beta"]),
        )
        encoded_abc, _ = study.encode_unit_score_metadata(exact_abc, "fp16")
        independent_score = study.complete_unit_scores(
            outputs.h2, outputs.h4, encoded_abc,
        )
        started = time.perf_counter()
        full_rows = study._hybrid_rows(
            outputs, gram, proxy, float(proxy_facts["beta"]),
            independent_score,
            {"layer": args.layer, "expert_id": args.expert},
            config, device=torch.device(args.device),
        )
        wall_seconds = time.perf_counter() - started
        per_family: dict[str, list[dict[str, object]]] = {}
        for record in full_rows:
            per_family.setdefault(str(record["selector_family"]), []).append({
                "budget_bpw": float(record["physical_budget_bpw"]),
                "pages": int(record["physical_pages"]),
                "recovery": float(record["recovery"]),
                "selected_seed": str(record["selected_seed"]),
                "incremental_seconds": float(record["selector_runtime_incremental_ms"]) / 1000.0,
                "total_including_seed_seconds": float(
                    record["selector_runtime_total_including_seed_ms"]
                ) / 1000.0,
            })
        full_hybrid = {
            "wall_seconds": wall_seconds,
            "rows": len(full_rows),
            "per_family": per_family,
            "scope": "one locked validation invocation; all three budgets; all four local-search seeds",
        }

    print(json.dumps({
        "layer": args.layer,
        "expert": args.expert,
        "router_rank": int(ranks[0]),
        "device": str(args.device),
        "primitive_gram_shape": list(trace.primitive_gram_shape),
        "primitive_gram_device_bytes": int(trace.primitive_gram_device_bytes),
        "candidate_evaluations": int(trace.candidate_evaluations),
        "gram_build_seconds": float(trace.gram_build_seconds),
        "gram_transfer_seconds": float(trace.gram_transfer_seconds),
        "greedy_seconds": float(trace.greedy_seconds),
        "snapshot_seconds": float(trace.snapshot_seconds),
        "total_seconds": float(trace.total_seconds),
        "snapshots": snapshots,
        "full_hybrid": full_hybrid,
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
