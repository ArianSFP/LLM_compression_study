#!/usr/bin/env python3
"""Regenerate incompatible D1 allocation groups on the exact PRO trajectory."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

from oracle_study.causal_control import (  # noqa: E402
    selector_and_execution_router_weights,
)
from oracle_study.d1_decode import (  # noqa: E402
    clone_decode_cache,
    mutation_safe_cached_autograd,
)
from oracle_study.d1_decode_tail import (  # noqa: E402
    COMPANION_D1_POLICY,
    DECODE_POLICIES,
    D1_PREFIX,
    FIXED_D1_POLICY,
    LOCAL_POLICY,
    PR13_POLICY,
    TOKEN_ORACLE_POLICY,
    sha256,
    validate_cached_decode_tail_config,
)
from oracle_study.d1_layer_slice import load_d1_layer_slice  # noqa: E402
from oracle_study.d1_route_objective import option_boundary_effects  # noqa: E402
import run_d1_slice_oracle_pilot as base  # noqa: E402


SCHEMA = "pr13_d1_tail_same_host_allocation_patch_v1"
GROUPS = "d1_tail_same_host_patch_groups.parquet"
EXPERTS = "d1_tail_same_host_patch_experts.parquet"
PARITY = "d1_tail_same_host_patch_parity.parquet"
DELTAS = "d1_tail_same_host_patch_deltas.npz"
FACTS = "d1_tail_same_host_patch_facts.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _load_capture(root: Path, layer: int) -> dict[str, np.ndarray]:
    facts = load_json(root / "d1_tail_same_host_allocation_capture_facts.json")
    if facts.get("schema") != "pr13_d1_tail_same_host_allocation_capture_v1":
        raise ValueError("unexpected same-host allocation capture schema")
    path = root / f"same_host_allocation_layer_{int(layer):02d}.npz"
    recorded = facts["files"].get(path.name)
    if not isinstance(recorded, Mapping) or str(recorded.get("sha256")) != sha256(path):
        raise RuntimeError("same-host allocation capture hash changed")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    required = {
        "request_id", "position", "group", "admitted", "hidden", "x",
        "residual", "routed", "router_logits", "router_scores", "router_ids",
        "d1_router_logits", "d1_router_scores", "d1_router_ids",
    }
    if not required.issubset(arrays):
        raise RuntimeError("same-host allocation capture fields are incomplete")
    if int(np.asarray(arrays["layer"]).reshape(-1)[0]) != int(layer):
        raise RuntimeError("same-host capture layer changed")
    return arrays


def _capture_record(capture: Mapping[str, np.ndarray], group: int) -> int:
    found = np.flatnonzero(
        np.asarray(capture["admitted"], bool)
        & (np.asarray(capture["group"], np.int64) == int(group))
    )
    if len(found) != 1:
        raise RuntimeError("same-host captured group identity is not unique")
    return int(found[0])


def _hidden_prefix(
    capture: Mapping[str, np.ndarray],
    request_id: str,
    position: int,
) -> np.ndarray:
    request = np.asarray(capture["request_id"]).astype(str)
    positions = np.asarray(capture["position"], np.int64)
    found = np.flatnonzero(
        (request == str(request_id)) & (positions < int(position))
    )
    found = found[np.argsort(positions[found], kind="stable")]
    if not np.array_equal(positions[found], np.arange(int(position))):
        raise RuntimeError("same-host hidden prefix is not contiguous")
    return np.asarray(capture["hidden"][found], np.float32)


def _groups(
    capture: Mapping[str, np.ndarray],
    global_groups: Sequence[int],
    layer: int,
    same_host_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    result = []
    records = []
    for local_group, global_group in enumerate(map(int, global_groups)):
        record = _capture_record(capture, global_group)
        selector, execution = selector_and_execution_router_weights(
            capture["router_scores"][record],
            historical_sum_atol=float(
                same_host_config["historical_router_sum_atol"]
            ),
            execution_sum_atol=float(
                same_host_config["same_host_execution_router_sum_atol"]
            ),
        )
        result.append({
            "group": local_group,
            "source_group": global_group,
            "request_id": str(capture["request_id"][record]),
            "position": int(capture["position"][record]),
            "layer": int(layer),
            "experts": np.asarray(capture["router_ids"][record], np.int64),
            "selector_weights": selector,
            "execution_weights": execution,
            "activation": np.asarray(capture["x"][record], np.float32),
        })
        records.append(record)
    return result, records


def _problem(
    model_slice: Any,
    capture: Mapping[str, np.ndarray],
    group: Mapping[str, Any],
    record: int,
    temperature: float,
    router_anchor_atol: float,
) -> tuple[Any, dict[str, Any], np.ndarray]:
    request_id = str(group["request_id"])
    position = int(group["position"])
    prefix_hidden = _hidden_prefix(capture, request_id, position)
    with torch.no_grad():
        prefix = model_slice.decode_prefix_cache(prefix_hidden)
        reconstructed = model_slice.compose_current_output(
            capture["residual"][record],
            capture["x"][record],
            capture["routed"][record],
        )
    reconstructed_np = reconstructed.detach().float().cpu().numpy().reshape(-1)
    reference_hidden = np.asarray(capture["hidden"][record], np.float32).reshape(-1)
    if not np.array_equal(reconstructed_np, reference_hidden):
        raise RuntimeError("same-host current-layer composition is not bit-identical")
    current = reconstructed.detach().reshape(1, 1, -1).clone().requires_grad_(True)
    with mutation_safe_cached_autograd():
        logits, _ = model_slice.next_router_logits_decode(
            current, clone_decode_cache(prefix), position,
        )
        values = logits[0, 0]
        values_np = values.detach().float().cpu().numpy()
        reference_logits = np.asarray(
            capture["d1_router_logits"][record], np.float32,
        )
        reference_ids = np.asarray(capture["d1_router_ids"][record], np.int64)
        observed_ids = torch.topk(values, 8).indices.detach().cpu().numpy()
        maximum = float(np.max(np.abs(values_np - reference_logits), initial=0.0))
        if maximum > float(router_anchor_atol):
            raise RuntimeError(
                "same-host D1 slice exceeds the frozen baseline anchor gate: "
                f"max_abs={maximum} > {router_anchor_atol}"
            )
        if not np.array_equal(observed_ids, reference_ids):
            raise RuntimeError("same-host D1 slice route IDs changed")
        selected_set = set(reference_ids.tolist())
        outsiders = np.asarray(sorted(
            (expert for expert in range(reference_logits.size) if expert not in selected_set),
            key=lambda expert: (-float(reference_logits[expert]), int(expert)),
        )[:8], np.int64)
        candidate_ids = np.concatenate((reference_ids[5:8], outsiders))
        sensitivities = np.zeros((values_np.size, current.shape[-1]), np.float64)
        started = time.perf_counter()
        for expert in candidate_ids.tolist():
            gradient = torch.autograd.grad(
                values[int(expert)],
                current,
                retain_graph=True,
                create_graph=False,
            )[0]
            sensitivities[int(expert)] = (
                gradient[0, 0].detach().float().cpu().numpy()
            )
        vjp_seconds = time.perf_counter() - started
    problem = base._exact_boundary_problem(
        reference_logits, sensitivities, reference_ids, float(temperature),
    )
    logit_anchor = reference_logits.astype(np.float64) - values_np.astype(np.float64)
    del logits, current, prefix
    return problem, {
        "schema": SCHEMA,
        "layer": int(group["layer"]),
        "group": int(group["source_group"]),
        "request_id": request_id,
        "position": position,
        "current_hidden_bit_identical": True,
        "d1_router_logits_bit_identical": bool(
            np.array_equal(values_np, reference_logits)
        ),
        "d1_router_max_abs": maximum,
        "d1_route_ids_order_equal": True,
        "baseline_router_logit_anchor_applied": True,
        "vjp_execution": "same_host_slice_with_exact_full_model_baseline_anchor",
        "exact_vjp_candidates": len(candidate_ids),
        "exact_vjp_seconds": vjp_seconds,
    }, logit_anchor


def _exact_replay(
    model_slice: Any,
    capture: Mapping[str, np.ndarray],
    group: Mapping[str, Any],
    record: int,
    geometry: Mapping[str, Any],
    trace: Any,
    logit_anchor: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    position = int(group["position"])
    prefix_hidden = _hidden_prefix(capture, str(group["request_id"]), position)
    delta = base._delta_for_trace(
        geometry,
        np.asarray(group["execution_weights"], np.float64),
        trace,
    )
    with torch.no_grad():
        prefix = model_slice.decode_prefix_cache(prefix_hidden)
        candidate_hidden = model_slice.compose_current_output(
            capture["residual"][record],
            capture["x"][record],
            capture["routed"][record],
            delta,
        ).reshape(1, 1, -1)
        candidate_logits, _, _, _ = model_slice.next_router_outputs_decode(
            candidate_hidden,
            clone_decode_cache(prefix),
            position,
        )
    anchored_logits = (
        candidate_logits[0, 0].float().cpu().numpy().astype(np.float64)
        + np.asarray(logit_anchor, np.float64)
    )
    anchored_ids = np.argsort(-anchored_logits, kind="stable")[:8]
    metrics = base._route_metrics(
        np.asarray(capture["d1_router_logits"][record], np.float32),
        anchored_logits,
        np.asarray(capture["d1_router_ids"][record], np.int64),
        anchored_ids,
    )
    return delta, metrics


def _source_rows(
    allocation_groups: pd.DataFrame,
    allocation_experts: pd.DataFrame,
    source_policy: str,
    local_group: int,
) -> tuple[pd.Series, pd.DataFrame]:
    group = allocation_groups[
        allocation_groups["policy"].eq(str(source_policy))
        & allocation_groups["group"].eq(int(local_group))
    ]
    experts = allocation_experts[
        allocation_experts["policy"].eq(str(source_policy))
        & allocation_experts["group"].eq(int(local_group))
    ].sort_values("router_rank", kind="stable")
    if len(group) != 1 or len(experts) != 8:
        raise RuntimeError("same-host allocation source rows are incomplete")
    return group.iloc[0], experts


def run_layer(
    args: argparse.Namespace,
    layer: int,
    global_groups: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    capture = _load_capture(args.capture_dir, int(layer))
    groups, records = _groups(
        capture, global_groups, int(layer), args.same_host_config_data,
    )
    factor_path = args.fit_dir / f"average_rate_factor_layer_{int(layer)}.npz"
    factor_sidecar = args.fit_dir / f"average_rate_factor_layer_{int(layer)}.json"
    if load_json(factor_sidecar)["sha256"] != sha256(factor_path):
        raise RuntimeError("PR13 factor sidecar hash mismatch")
    arrays = dict(np.load(factor_path, allow_pickle=False))
    proxy = np.asarray(arrays["proxy"], np.float32)
    beta = float(np.asarray(arrays["beta"]).reshape(-1)[0])
    prepare_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        trees=args.trees,
        workers=int(args.workers),
    )
    experts = base._prepare_experts(
        prepare_args,
        int(layer),
        groups,
        arrays,
        proxy,
        beta,
        args.pr13_config_data,
    )
    coarse = base._build_geometries(
        groups,
        experts,
        proxy,
        beta,
        args.pr13_config_data,
        int(args.workers),
        int(layer),
    )
    del experts
    gc.collect()

    model_slice = load_d1_layer_slice(
        args.checkpoint, int(layer), device=args.device,
    )
    problems = []
    parity_rows = []
    logit_anchors = []
    for group, record in zip(groups, records):
        problem, parity, logit_anchor = _problem(
            model_slice,
            capture,
            group,
            record,
            float(args.temperature),
            float(args.slice_router_anchor_max_abs),
        )
        problems.append(problem)
        parity_rows.append(parity)
        logit_anchors.append(logit_anchor)

    output_groups = []
    output_experts = []
    for rate in map(int, args.tail_config_data["page_caps"]):
        refined = base._refine_geometries(
            coarse,
            groups,
            proxy,
            beta,
            args.pr13_config_data,
            int(rate),
            int(args.workers),
            int(layer),
        )
        effects = [
            option_boundary_effects(
                geometry["option_deltas"],
                np.asarray(group["execution_weights"], np.float64),
                problem,
            )
            for geometry, group, problem in zip(refined, groups, problems)
        ]
        selections, group_rows, expert_rows = base._allocation_policies(
            groups,
            refined,
            {float(args.temperature): problems},
            effects,
            [int(rate)],
            list(map(float, args.eta)),
            args.pr13_config_data,
            base.POLICY_LOCAL_REFINED,
        )
        group_frame = pd.DataFrame(group_rows)
        expert_frame = pd.DataFrame(expert_rows)
        source_metrics: dict[tuple[int, str], dict[str, Any]] = {}
        source_deltas: dict[tuple[int, str], np.ndarray] = {}
        d1_sources = sorted(
            policy for policy, selected_rate in selections
            if int(selected_rate) == int(rate) and str(policy).startswith(D1_PREFIX)
        )
        for local_group, (group, record, geometry) in enumerate(
            zip(groups, records, refined)
        ):
            for source_policy in sorted(
                {base.POLICY_REGENERATED, base.POLICY_LOCAL_REFINED, *d1_sources}
            ):
                trace = selections[(source_policy, int(rate))][local_group]
                delta, metrics = _exact_replay(
                    model_slice, capture, group, record, geometry, trace,
                    logit_anchors[local_group],
                )
                source_metrics[(local_group, source_policy)] = metrics
                source_deltas[(local_group, source_policy)] = delta
            token_source = min(
                d1_sources,
                key=lambda policy: (
                    int(source_metrics[(local_group, policy)]["exact_d1_crossed"]),
                    int(source_metrics[(local_group, policy)]["membership_pairs_changed"]),
                    float(source_metrics[(local_group, policy)]["routing_mass_lost"]),
                    float(_source_rows(
                        group_frame, expert_frame, policy, local_group,
                    )[0]["local_qenergy_damage"]),
                    int(_source_rows(
                        group_frame, expert_frame, policy, local_group,
                    )[0]["selected_group_pages"]),
                    policy,
                ),
            )
            frozen = args.tail_config_data["d1_source_policy_by_rate"][str(rate)]
            sources = {
                PR13_POLICY: base.POLICY_REGENERATED,
                LOCAL_POLICY: base.POLICY_LOCAL_REFINED,
                FIXED_D1_POLICY: str(frozen["fixed"]),
                COMPANION_D1_POLICY: str(frozen["companion"]),
                TOKEN_ORACLE_POLICY: token_source,
            }
            if tuple(sources) != DECODE_POLICIES:
                raise RuntimeError("same-host patch policy order changed")
            for policy, source_policy in sources.items():
                group_source, expert_source = _source_rows(
                    group_frame, expert_frame, source_policy, local_group,
                )
                metrics = source_metrics[(local_group, source_policy)]
                output_groups.append({
                    "schema": SCHEMA,
                    "layer": int(layer),
                    "rate_pages_per_expert": int(rate),
                    "group": int(group["source_group"]),
                    "request_id": str(group["request_id"]),
                    "position": int(group["position"]),
                    "policy": policy,
                    "source_policy": source_policy,
                    "selected_group_pages": int(group_source["selected_group_pages"]),
                    "local_qenergy_damage": float(group_source["local_qenergy_damage"]),
                    "_selected_delta": source_deltas[(local_group, source_policy)],
                    **metrics,
                })
                for row in expert_source.itertuples(index=False):
                    output_experts.append({
                        "schema": SCHEMA,
                        "layer": int(layer),
                        "rate_pages_per_expert": int(rate),
                        "group": int(group["source_group"]),
                        "request_id": str(group["request_id"]),
                        "position": int(group["position"]),
                        "policy": policy,
                        "source_policy": source_policy,
                        "router_rank": int(row.router_rank),
                        "expert_id": int(row.expert_id),
                        "selector_router_weight": float(row.selector_router_weight),
                        "execution_router_weight": float(row.execution_router_weight),
                        "selected_option_index": int(row.selected_option_index),
                        "selected_pages": int(row.selected_pages),
                        "selected_states": str(row.selected_states),
                    })
        del refined, selections
        gc.collect()
    del model_slice, arrays, coarse
    torch.cuda.empty_cache()
    gc.collect()
    return output_groups, output_experts, parity_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-config", type=Path, required=True)
    parser.add_argument("--pr13-config", type=Path, required=True)
    parser.add_argument("--same-host-config", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eta", type=float, nargs="+", required=True)
    parser.add_argument("--temperature", type=float, default=0.0625)
    parser.add_argument("--slice-router-anchor-max-abs", type=float, default=0.0625)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.tail_config_data = load_json(args.tail_config)
    args.pr13_config_data = load_json(args.pr13_config)
    args.same_host_config_data = load_json(args.same_host_config)
    validate_cached_decode_tail_config(args.tail_config_data)
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must lie in [1,32]")
    if args.temperature <= 0.0 or any(value < 0.0 for value in args.eta):
        raise ValueError("D1 temperature/eta grid is invalid")
    if not (0.0 <= args.slice_router_anchor_max_abs <= 0.125):
        raise ValueError("slice router anchor limit must lie in [0,0.125]")

    audit = pd.read_parquet(args.route_audit)
    mismatch = audit[~audit["route_set_equal"]].copy()
    if len(mismatch) == 0:
        raise RuntimeError("same-host allocation patch has no mismatched groups")
    by_layer = {
        int(layer): sorted(frame["group"].astype(int).unique().tolist())
        for layer, frame in mismatch.groupby("layer", sort=True)
    }
    started = time.perf_counter()
    group_rows = []
    expert_rows = []
    parity_rows = []
    for layer, global_groups in sorted(by_layer.items()):
        groups, experts, parity = run_layer(args, layer, global_groups)
        group_rows.extend(groups)
        expert_rows.extend(experts)
        parity_rows.extend(parity)
        print(
            f"[same-host patch] layer={layer} groups={len(global_groups)} complete",
            flush=True,
        )

    groups_frame = pd.DataFrame(group_rows).sort_values(
        ["layer", "rate_pages_per_expert", "group", "policy"],
        kind="stable",
    ).reset_index(drop=True)
    experts_frame = pd.DataFrame(expert_rows).sort_values(
        ["layer", "rate_pages_per_expert", "group", "policy", "router_rank"],
        kind="stable",
    ).reset_index(drop=True)
    parity_frame = pd.DataFrame(parity_rows).sort_values(
        ["layer", "group"], kind="stable",
    ).reset_index(drop=True)
    selected_deltas = np.stack(groups_frame.pop("_selected_delta").tolist()).astype(
        np.float32,
    )
    expected_groups = len(mismatch) * len(args.tail_config_data["page_caps"]) * len(DECODE_POLICIES)
    if len(groups_frame) != expected_groups or len(experts_frame) != 8 * expected_groups:
        raise RuntimeError("same-host allocation patch output grid is incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    base.atomic_npz(
        args.output / DELTAS,
        layer=groups_frame["layer"].to_numpy(np.int64),
        rate_pages_per_expert=groups_frame["rate_pages_per_expert"].to_numpy(np.int64),
        group=groups_frame["group"].to_numpy(np.int64),
        policy=np.asarray(
            groups_frame["policy"].astype(str).tolist(), dtype="<U64",
        ),
        selected_deltas=selected_deltas,
    )
    base.atomic_parquet(args.output / GROUPS, groups_frame)
    base.atomic_parquet(args.output / EXPERTS, experts_frame)
    base.atomic_parquet(args.output / PARITY, parity_frame)
    outputs = {
        name: {
            "sha256": sha256(args.output / name),
            "bytes": (args.output / name).stat().st_size,
            "rows": len(frame),
        }
        for name, frame in (
            (GROUPS, groups_frame),
            (EXPERTS, experts_frame),
            (PARITY, parity_frame),
        )
    }
    outputs[DELTAS] = {
        "sha256": sha256(args.output / DELTAS),
        "bytes": (args.output / DELTAS).stat().st_size,
        "rows": len(selected_deltas),
    }
    facts = {
        "completed": True,
        "schema": SCHEMA,
        "tail_config_sha256": sha256(args.tail_config),
        "pr13_config_sha256": sha256(args.pr13_config),
        "same_host_config_sha256": sha256(args.same_host_config),
        "capture_facts_sha256": sha256(
            args.capture_dir / "d1_tail_same_host_allocation_capture_facts.json"
        ),
        "route_audit_sha256": sha256(args.route_audit),
        "mismatched_groups": len(mismatch),
        "groups_by_layer": {str(key): value for key, value in sorted(by_layer.items())},
        "rates": list(map(int, args.tail_config_data["page_caps"])),
        "policies": list(DECODE_POLICIES),
        "eta": list(map(float, args.eta)),
        "temperature": float(args.temperature),
        "outputs": outputs,
        "allocation_patch_gate_passed": bool(
            parity_frame["current_hidden_bit_identical"].all()
            and parity_frame["d1_route_ids_order_equal"].all()
            and float(parity_frame["d1_router_max_abs"].max())
            <= float(args.slice_router_anchor_max_abs)
        ),
        "all_current_hidden_bit_identical": bool(
            parity_frame["current_hidden_bit_identical"].all()
        ),
        "all_raw_slice_router_logits_bit_identical": bool(
            parity_frame["d1_router_logits_bit_identical"].all()
        ),
        "all_slice_route_ids_order_equal": bool(
            parity_frame["d1_route_ids_order_equal"].all()
        ),
        "maximum_raw_slice_router_max_abs": float(
            parity_frame["d1_router_max_abs"].max()
        ),
        "slice_router_anchor_max_abs": float(args.slice_router_anchor_max_abs),
        "baseline_router_logit_anchor_applied": True,
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "exact_q4_route_runtime_input": False,
        "oracle_scope": "same_host_tail_allocation_compatibility_only",
        "test_rows_admitted_or_used": False,
    }
    base.atomic_json(args.output / FACTS, facts)
    print(json.dumps(facts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
