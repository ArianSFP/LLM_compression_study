#!/usr/bin/env python3
"""Rapid hard-layer screen for globally aligned JL interaction fields."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

import run_average_rate_allocation as pr13
import run_half_bpw_compact_field as compact
import run_half_bpw_joint_field as teacher
import run_sparse_streaming_study as base
from oracle_study.low_rate_joint_field import combine_split_interaction_fields
from oracle_study.shared_jl_field import (
    calibrated_shared_factor,
    quantized_shared_factor_with_pair_calibration,
    rademacher_projection,
)
from oracle_study.split_interaction_field import (
    SPLIT_STATE_PAGE_COSTS,
    build_split_interaction_field,
    split_coordinate_descent,
    split_local_search,
    split_projection_responses,
)


PR13_FACTOR_BYTES = 6148
PR13_GROUP_BYTES = 8 * (3084 + PR13_FACTOR_BYTES)
PR13_GROUP_MACS = 79_712_256
EXACT_GROUP_BYTES = 8 * (3084 + 2 * 512 * 2052 * 8)
EXACT_FROZEN_MACS = 2_058_007_296
POLICY_SPECS = (
    ("shared_jl_rank32_int4", 32, "int4_per_row"),
    ("shared_jl_rank64_int4", 64, "int4_per_row"),
    ("shared_jl_rank64_int8", 64, "int8_per_row"),
    ("shared_jl_rank64_fp32_control", 64, None),
    ("shared_jl_rank128_int4", 128, "int4_per_row"),
    ("shared_jl_rank256_ternary", 256, "ternary_per_row"),
    ("shared_jl_rank512_sign", 512, "sign_per_row"),
)
POLICIES = (
    "frozen_pr13_exact_combined_witness",
    *(spec[0] for spec in POLICY_SPECS),
    "frozen_joint_exact_qmetric_oracle",
)
_CONTEXT = None


def _worker(task: int):
    if _CONTEXT is None:
        raise RuntimeError("shared-JL probe worker lacks context")
    group = _CONTEXT["groups"][int(task)]
    weights = np.asarray(group["router_weights"], np.float64)
    activation = np.asarray(_CONTEXT["local"]["x"][int(group["record"])], np.float64)
    ids = tuple(map(int, group["experts"]))
    responses = tuple(
        split_projection_responses(
            _CONTEXT["experts"][expert]["q2"],
            _CONTEXT["experts"][expert]["q4"], activation,
        )
        for expert in ids
    )
    identity = tuple(group[name] for name in teacher.GROUP_IDENTITY)
    witness = np.asarray(_CONTEXT["witnesses"][identity], np.int64)
    exact = np.asarray(_CONTEXT["teacher_states"][identity], np.int64)
    budget = int(_CONTEXT["config"]["group_page_budget"])
    base_states = np.zeros(4096, np.int64)
    base_damage = teacher._exact_group_damage(
        responses, weights, base_states, _CONTEXT["proxy"], _CONTEXT["beta"],
    )
    records = [(POLICIES[0], witness, 0, 0, PR13_GROUP_MACS, PR13_GROUP_BYTES, 0.0)]
    for policy, _, _ in POLICY_SPECS:
        fields = tuple(
            build_split_interaction_field(
                _CONTEXT["experts"][expert]["factors"][policy],
                responses[index].hidden,
                _CONTEXT["experts"][expert]["abc"],
            )
            for index, expert in enumerate(ids)
        )
        joint = combine_split_interaction_fields(fields, weights)
        started = time.perf_counter()
        coordinate = split_coordinate_descent(
            joint, witness, budget,
            max_sweeps=int(_CONTEXT["coordinate_sweeps"]),
        )
        local = split_local_search(
            joint, coordinate.states, budget,
            shortlist_size=int(_CONTEXT["config"]["local_shortlist"]),
            max_swap_units=int(_CONTEXT["config"]["local_swap_units"]),
            max_passes=int(_CONTEXT["local_max_passes"]),
        )
        wall = time.perf_counter() - started
        factor_bytes = int(fields[0].factor.payload_bytes)
        geometry = 8 * (3084 + PR13_FACTOR_BYTES + factor_bytes)
        refinement_macs = teacher._solver_macs(
            joint.rank, joint.units, coordinate.sweeps, local.passes,
            int(_CONTEXT["config"]["local_shortlist"]),
        )
        records.append((
            policy, local.states, coordinate.sweeps, local.passes,
            PR13_GROUP_MACS + int(refinement_macs), geometry, wall,
        ))
    records.append((
        POLICIES[-1], exact, 0, 0, EXACT_FROZEN_MACS, EXACT_GROUP_BYTES, 0.0,
    ))
    rows = []
    for policy, states, sweeps, passes, macs, geometry, wall in records:
        pages = int(SPLIT_STATE_PAGE_COSTS[np.asarray(states, np.int64)].sum())
        damage = teacher._exact_group_damage(
            responses, weights, states, _CONTEXT["proxy"], _CONTEXT["beta"],
        )
        total_bytes = int(geometry) + pages * 512
        rows.append({
            **{name: group[name] for name in teacher.GROUP_IDENTITY},
            "sequence_id": group["sequence_id"],
            "allocation_policy": policy,
            "actual_group_pages": pages,
            "average_actual_streamed_bpw": pages / (8.0 * 768.0),
            "group_recovery": 1.0 - damage / base_damage,
            "group_exact_qenergy_damage": damage,
            "group_base_qenergy_damage": base_damage,
            "logical_geometry_bytes_read": int(geometry),
            "total_geometry_and_correction_bytes": total_bytes,
            "average_actual_total_bpw": total_bytes / 3_145_728.0,
            "selector_compute_macs": int(macs),
            "compute_ratio_vs_pr13": int(macs) / PR13_GROUP_MACS,
            "geometry_ratio_vs_pr13": int(geometry) / PR13_GROUP_BYTES,
            "coordinate_sweeps": int(sweeps),
            "local_passes": int(passes),
            "selector_wall_seconds": float(wall),
            "within_ten_x_pr13_geometry": int(geometry) <= 10 * PR13_GROUP_BYTES,
            "within_ten_x_pr13_compute": int(macs) <= 10 * PR13_GROUP_MACS,
            "exploratory_probe": True,
            "promotion_eligible": False,
        })
    return int(task), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--pr13-fit-dir", type=Path, required=True)
    parser.add_argument("--pr13-validation-dir", type=Path, required=True)
    parser.add_argument("--pr12-config", type=Path, required=True)
    parser.add_argument("--pr12-dir", type=Path, required=True)
    parser.add_argument("--pr10-config", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--compact-fit-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--exact-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 20])
    parser.add_argument("--groups-per-layer", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--coordinate-sweeps", type=int)
    parser.add_argument("--local-max-passes", type=int)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--projection-method",
        choices=(
            "rademacher", "active_weight_pca", "decision_plus_pca",
            "supervised_residual_pca",
        ),
        default="rademacher",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("shared-JL probe output must be a new directory")
    config = json.loads(args.config.read_text())
    compact._validate_config(config)
    compact._base_inputs(args, config)
    teacher_states, _ = compact._verify_teacher(config, args.teacher_dir)
    witnesses, _ = teacher._load_witnesses(config, args.pr13_validation_dir)
    data = pr13.prior.load_capture_admitted(args.exact_captures, ["train", "validation"])
    groups = compact._plan_split(data, config, "validation")
    selected = []
    for layer in args.layers:
        selected.extend(
            [group for group in groups if int(group["layer"]) == int(layer)]
            [: int(args.groups_per_layer)]
        )
    if len(selected) != len(args.layers) * int(args.groups_per_layer):
        raise RuntimeError("shared-JL probe group count changed")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tree_records = json.loads(args.trees.read_text())
    trees = {name: pr13.tree_from_record(tree_records[name]) for name in pr13.PROJECTIONS}
    all_rows, diagnostics = [], []
    maximum_tail = max(rank for _, rank, _ in POLICY_SPECS) - 4
    for layer in args.layers:
        layer_groups = [group for group in selected if int(group["layer"]) == int(layer)]
        local = pr13.prior.layer_view(data, int(layer))
        proxy, proxy_facts = base.proxy_gradients(
            local["xplus"], [local[f"h{h}_router_logits"] for h in range(1, 5)],
            local["split"],
        )
        if proxy.shape[1] != 4:
            raise RuntimeError("shared-JL probe requires the frozen rank-4 proxy")
        beta = float(proxy_facts["beta"])
        root_beta = np.sqrt(beta)
        active = sorted({int(expert) for group in layer_groups for expert in group["experts"]})
        decoded_by_expert = {
            expert: pr13.prior.decode_expert(
                args.checkpoint, index, trees, int(layer), expert,
            )
            for expert in active
        }
        projection = None
        if args.projection_method == "rademacher":
            projection = rademacher_projection(
                2048, maximum_tail, int(args.seed) + int(layer),
            )
        elif args.projection_method != "supervised_residual_pca":
            import torch

            training_rows = np.concatenate([
                np.concatenate((
                    np.asarray(decoded_by_expert[expert]["down"][0].T, np.float32),
                    np.asarray(decoded_by_expert[expert]["down"][2].T, np.float32),
                ))
                for expert in active
            ])
            norms = np.linalg.norm(training_rows, axis=1)
            training_rows /= np.maximum(
                norms[:, None], np.finfo(np.float32).tiny,
            )
            generator_state = torch.random.get_rng_state()
            torch.manual_seed(int(args.seed) + int(layer))
            tensor = torch.from_numpy(training_rows).to(args.device)
            _, _, right = torch.pca_lowrank(
                tensor, q=maximum_tail, center=False, niter=3,
            )
            projection = right.cpu().numpy().astype(np.float32)
            torch.random.set_rng_state(generator_state)
            del tensor, training_rows
        decision_components = None
        supervised_basis_energy = None
        if args.projection_method == "supervised_residual_pca":
            training_groups = [
                group for group in compact._plan_split(data, config, "train")
                if int(group["layer"]) == int(layer)
            ]
            training_active = sorted({
                int(expert) for group in training_groups for expert in group["experts"]
            })
            for expert in training_active:
                if expert not in decoded_by_expert:
                    decoded_by_expert[expert] = pr13.prior.decode_expert(
                        args.checkpoint, index, trees, int(layer), expert,
                    )
            training_experts = {}
            for expert in training_active:
                decoded = decoded_by_expert[expert]
                q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
                q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
                training_experts[expert] = {
                    "q2": q2, "q4": q4,
                    "abc": pr13.unit_score_metadata(
                        q2[2], q4[2], proxy=proxy, beta=beta,
                    ),
                }
            compact._FIT_CONTEXT = {
                "groups": tuple(training_groups), "local": local,
                "experts": training_experts, "proxy": proxy,
                "beta": beta, "config": config,
            }
            pending_trajectory = {}
            with ProcessPoolExecutor(
                max_workers=min(int(config["fit_workers"]), len(training_groups)),
                mp_context=mp.get_context(str(config["worker_start_method"])),
            ) as executor:
                futures = {
                    executor.submit(compact._fit_teacher_group, task): task
                    for task in range(len(training_groups))
                }
                for future in as_completed(futures):
                    task, residuals, _ = future.result()
                    pending_trajectory[task] = residuals
            compact._FIT_CONTEXT = None
            trajectory = np.concatenate([
                pending_trajectory[task] for task in range(len(training_groups))
            ])
            basis = compact.fit_shared_decision_basis(
                trajectory, proxy, beta, max(rank for _, rank, _ in POLICY_SPECS),
                normalize_rows=True,
            )
            decision_components = np.asarray(basis.components, np.float64)
            supervised_basis_energy = float(basis.explained_energy)
        elif args.projection_method == "decision_plus_pca":
            if args.compact_fit_dir is None:
                raise RuntimeError("decision-plus-PCA requires --compact-fit-dir")
            with np.load(
                args.compact_fit_dir / f"compact_half_bpw_factor_layer_{layer}.npz",
                allow_pickle=False,
            ) as arrays:
                decision = np.asarray(arrays["basis_components"], np.float64)
            if decision.shape != (2052, 16):
                raise RuntimeError("frozen decision basis shape changed")
            augmented = np.concatenate((
                np.asarray(projection[:, :112], np.float64),
                np.zeros((4, 112), np.float64),
            ))
            augmented -= decision @ (decision.T @ augmented)
            tail_basis, _ = np.linalg.qr(augmented)
            decision_components = np.concatenate(
                (decision, tail_basis[:, :112]), axis=1,
            )
        experts = {}
        shrink_by_policy = {policy: [] for policy, _, _ in POLICY_SPECS}
        calibration_by_policy = {policy: [] for policy, _, _ in POLICY_SPECS}
        for expert in active:
            decoded = decoded_by_expert[expert]
            q2 = tuple(decoded[name][0] for name in pr13.PROJECTIONS)
            q4 = tuple(decoded[name][2] for name in pr13.PROJECTIONS)
            down2, down4 = np.asarray(q2[2].T), np.asarray(q4[2].T)
            abc = pr13.unit_score_metadata(q2[2], q4[2], proxy=proxy, beta=beta)
            if decision_components is None:
                maximum_low, maximum_high = down2 @ projection, down4 @ projection
                proxy_low = root_beta * (down2 @ proxy)
                proxy_high = root_beta * (down4 @ proxy)
            else:
                feature_low = np.concatenate(
                    (down2, root_beta * (down2 @ proxy)), axis=1,
                )
                feature_high = np.concatenate(
                    (down4, root_beta * (down4 @ proxy)), axis=1,
                )
                maximum_low = feature_low @ decision_components
                maximum_high = feature_high @ decision_components
                proxy_low = proxy_high = np.empty((512, 0), np.float64)
            factors = {}
            for policy, rank, encoding in POLICY_SPECS:
                tail = rank if decision_components is not None else rank - 4
                normalization = (
                    1.0 if decision_components is not None
                    else np.sqrt(maximum_tail / tail)
                )
                result = calibrated_shared_factor(
                    maximum_high[:, :tail] * normalization,
                    maximum_low[:, :tail] * normalization,
                    proxy_high, proxy_low, abc,
                    encoding=(
                        None if encoding in {"ternary_per_row", "sign_per_row"}
                        else encoding
                    ),
                )
                if encoding in {"ternary_per_row", "sign_per_row"}:
                    result = quantized_shared_factor_with_pair_calibration(
                        result.factor, abc, encoding=encoding,
                    )
                factors[policy] = result.factor
                shrink_by_policy[policy].append(result.self_safe_scale)
                calibration_by_policy[policy].append(result.maximum_calibration_error)
            experts[expert] = {"q2": q2, "q4": q4, "abc": abc, "factors": factors}
        global _CONTEXT
        _CONTEXT = {
            "groups": tuple(layer_groups), "local": local, "experts": experts,
            "proxy": proxy, "beta": beta, "config": config,
            "witnesses": witnesses, "teacher_states": teacher_states,
            "coordinate_sweeps": (
                int(config["coordinate_sweeps"])
                if args.coordinate_sweeps is None else int(args.coordinate_sweeps)
            ),
            "local_max_passes": (
                int(config["local_max_passes"])
                if args.local_max_passes is None else int(args.local_max_passes)
            ),
        }
        pending = {}
        with ProcessPoolExecutor(
            max_workers=min(int(args.workers), len(layer_groups)),
            mp_context=mp.get_context("fork"),
        ) as executor:
            futures = {executor.submit(_worker, task): task for task in range(len(layer_groups))}
            for future in as_completed(futures):
                task, rows = future.result()
                pending[task] = rows
        for task in range(len(layer_groups)):
            all_rows.extend(pending[task])
        _CONTEXT = None
        for policy, _, _ in POLICY_SPECS:
            diagnostics.append({
                "layer": int(layer), "allocation_policy": policy,
                "active_experts": len(active),
                "median_self_safe_scale": float(np.median(shrink_by_policy[policy])),
                "supervised_basis_explained_energy": supervised_basis_energy,
                "minimum_self_safe_scale": float(np.min(shrink_by_policy[policy])),
                "maximum_prequant_calibration_error": float(np.max(calibration_by_policy[policy])),
            })
    frame = pd.DataFrame(all_rows)
    if len(frame) != len(selected) * len(POLICIES):
        raise RuntimeError("shared-JL probe row grid changed")
    summaries = []
    for policy, part in frame.groupby("allocation_policy", sort=False):
        values = part["group_recovery"].to_numpy(float)
        summaries.append({
            "allocation_policy": policy,
            "groups": len(part),
            "p10_recovery": float(np.quantile(values, 0.1)),
            "median_recovery": float(np.median(values)),
            "p90_recovery": float(np.quantile(values, 0.9)),
            "median_geometry_bytes_read": float(np.median(part["logical_geometry_bytes_read"])),
            "median_selector_compute_macs": float(np.median(part["selector_compute_macs"])),
            "median_compute_ratio_vs_pr13": float(np.median(part["compute_ratio_vs_pr13"])),
            "median_geometry_ratio_vs_pr13": float(np.median(part["geometry_ratio_vs_pr13"])),
            "median_selector_wall_seconds": float(np.median(part["selector_wall_seconds"])),
        })
    args.output.mkdir(parents=True)
    frame.to_parquet(args.output / "shared_jl_probe_frontier.parquet", index=False)
    compact._atomic_json(args.output / "shared_jl_probe_summary.json", {
        "completed": True,
        "exploratory_probe": True,
        "promotion_eligible": False,
        "layers": list(map(int, args.layers)),
        "groups_per_layer": int(args.groups_per_layer),
        "seed": int(args.seed),
        "projection_method": str(args.projection_method),
        "policies": list(POLICIES),
        "summaries": summaries,
        "diagnostics": diagnostics,
        "runner_sha256": compact._sha256(Path(__file__)),
        "core_sha256": compact._sha256(EXPERIMENT / "src/oracle_study/shared_jl_field.py"),
        "no_test_scientific_rows_admitted_or_used": True,
    })
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
