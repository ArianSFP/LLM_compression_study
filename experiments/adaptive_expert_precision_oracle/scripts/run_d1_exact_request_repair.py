#!/usr/bin/env python3
"""Exact request-coupled repair over already-generated D1 finalists.

The primary D1 allocator linearizes each token independently. During prefill,
however, errors at earlier positions also enter the causal next-layer mixer.
This validation-only repair searches mixtures of complete D1 finalist options
and scores each mixture by exact paired D1 replay. It is an oracle reranker,
never a proposed runtime component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_layer_slice import load_d1_layer_slice  # noqa: E402
from run_d1_slice_oracle_pilot import (  # noqa: E402
    SCHEMA,
    _capture_path,
    _load_capture,
    _route_metrics,
    atomic_json,
    atomic_npz,
    atomic_parquet,
    sha256,
)


REPAIR_SCHEMA = "pr13_d1_exact_request_repair_v1"
REPAIRED = "d1_exact_request_group_repaired"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--layer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coordinate-sweeps", type=int, default=2)
    parser.add_argument("--pair-passes", type=int, default=2)
    parser.add_argument("--max-pair-evaluations", type=int, default=10000)
    return parser.parse_args()


def _policy_key(policy: str, rate: int) -> str:
    return f"{policy}__rate_{int(rate)}"


def _candidate_policies(
    deltas: Mapping[str, np.ndarray], rate: int,
) -> tuple[str, ...]:
    suffix = f"__rate_{int(rate)}"
    policies = sorted(
        key[: -len(suffix)]
        for key in deltas
        if key.endswith(suffix)
        and (
            key.startswith("d1_strict_")
            or key.startswith("exact_combined_local_")
        )
    )
    if not policies:
        raise RuntimeError("no D1 finalist policies were stored")
    return tuple(policies)


def _identity_rows(
    allocation: pd.DataFrame, policy: str, rate: int,
) -> pd.DataFrame:
    result = allocation[
        allocation["policy"].eq(policy)
        & allocation["rate_pages_per_expert"].eq(int(rate))
    ][["group", "request_id", "position"]].copy()
    if result["group"].duplicated().any():
        raise RuntimeError("allocation group identity is not unique")
    return result.sort_values("group", kind="stable")


def _allocation_lookup(
    allocation: pd.DataFrame, rate: int,
) -> dict[tuple[int, str], pd.Series]:
    rows = allocation[allocation["rate_pages_per_expert"].eq(int(rate))]
    return {
        (int(row.group), str(row.policy)): row
        for row in rows.itertuples(index=False)
    }


class ExactRequestEvaluator:
    def __init__(
        self,
        *,
        model_slice: Any,
        capture: Mapping[str, np.ndarray],
        request_groups: list[dict[str, Any]],
        deltas: Mapping[str, np.ndarray],
        rate: int,
        allocation_lookup: Mapping[tuple[int, str], Any],
    ) -> None:
        self.model_slice = model_slice
        self.capture = capture
        self.request_groups = request_groups
        self.deltas = deltas
        self.rate = int(rate)
        self.allocation_lookup = allocation_lookup
        with torch.inference_mode():
            hidden = model_slice.compose_current_output(
                capture["residual"], capture["x"], capture["routed"],
            )
            logits, _, ids = model_slice.next_router_outputs(
                hidden, capture["attention_mask"],
            )
        self.baseline_logits = logits[0].detach().float().cpu().numpy()
        self.baseline_ids = ids[0].detach().cpu().numpy()
        self.evaluations = 0

    def evaluate(
        self, choice: Mapping[int, str],
    ) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
        request_delta = np.zeros_like(self.capture["routed"], np.float32)
        for group in self.request_groups:
            index = int(group["group"])
            policy = str(choice[index])
            request_delta[int(group["position"])] = self.deltas[
                _policy_key(policy, self.rate)
            ][index]
        with torch.inference_mode():
            hidden = self.model_slice.compose_current_output(
                self.capture["residual"],
                self.capture["x"],
                self.capture["routed"],
                request_delta,
            )
            logits, _, ids = self.model_slice.next_router_outputs(
                hidden, self.capture["attention_mask"],
            )
        self.evaluations += 1
        candidate_logits = logits[0].detach().float().cpu().numpy()
        candidate_ids = ids[0].detach().cpu().numpy()
        rows = []
        local = 0.0
        pages = 0
        for group in self.request_groups:
            index = int(group["group"])
            position = int(group["position"])
            policy = str(choice[index])
            allocation = self.allocation_lookup[(index, policy)]
            local += float(allocation.local_qenergy_damage)
            pages += int(allocation.selected_group_pages)
            rows.append({
                "schema": SCHEMA,
                "layer": int(group["layer"]),
                "next_layer": int(group["layer"]) + 1,
                "group": index,
                "request_id": str(group["request_id"]),
                "position": position,
                "policy": REPAIRED,
                "rate_pages_per_expert": self.rate,
                "local_qenergy_damage": float(allocation.local_qenergy_damage),
                **_route_metrics(
                    self.baseline_logits[position],
                    candidate_logits[position],
                    self.baseline_ids[position],
                    candidate_ids[position],
                ),
            })
        crossings = sum(bool(row["exact_d1_crossed"]) for row in rows)
        changed = sum(int(row["membership_pairs_changed"]) for row in rows)
        mass = sum(float(row["routing_mass_lost"]) for row in rows)
        signature = tuple(str(choice[int(group["group"])]) for group in self.request_groups)
        score = (crossings, changed, local, mass, pages, signature)
        return score, rows


def _repair_request(
    evaluator: ExactRequestEvaluator,
    group_policies: Mapping[int, tuple[str, ...]],
    *,
    coordinate_sweeps: int,
    pair_passes: int,
    max_pair_evaluations: int,
) -> tuple[dict[int, str], tuple[Any, ...], list[dict[str, Any]], dict[str, int]]:
    groups = [int(row["group"]) for row in evaluator.request_groups]
    global_candidates = sorted(set.intersection(*(
        set(group_policies[group]) for group in groups
    )))
    if not global_candidates:
        raise RuntimeError("request groups have no shared D1 finalist policy")
    best_choice: dict[int, str] | None = None
    best_score: tuple[Any, ...] | None = None
    best_rows: list[dict[str, Any]] | None = None
    for policy in global_candidates:
        choice = {group: policy for group in groups}
        score, rows = evaluator.evaluate(choice)
        if best_score is None or score < best_score:
            best_choice, best_score, best_rows = choice, score, rows
    assert best_choice is not None and best_score is not None and best_rows is not None
    seed_score = best_score

    coordinate_evaluations = 0
    for _ in range(int(coordinate_sweeps)):
        changed = False
        for group in groups:
            local_choice = dict(best_choice)
            local_score, local_rows = best_score, best_rows
            for policy in group_policies[group]:
                if policy == best_choice[group]:
                    continue
                candidate = dict(best_choice)
                candidate[group] = policy
                score, rows = evaluator.evaluate(candidate)
                coordinate_evaluations += 1
                if score < local_score:
                    local_choice, local_score, local_rows = candidate, score, rows
            if local_score < best_score:
                best_choice, best_score, best_rows = local_choice, local_score, local_rows
                changed = True
        if not changed:
            break

    pair_evaluations = 0
    for _ in range(int(pair_passes)):
        if int(best_score[0]) == 0:
            break
        crossed_positions = [
            int(row["position"]) for row in best_rows if row["exact_d1_crossed"]
        ]
        earliest = min(crossed_positions)
        eligible = [
            int(row["group"])
            for row in evaluator.request_groups
            if int(row["position"]) <= earliest
        ]
        pair_choice, pair_score, pair_rows = best_choice, best_score, best_rows
        stop = False
        for left_offset, left in enumerate(eligible):
            for right in eligible[left_offset + 1 :]:
                for left_policy in group_policies[left]:
                    if left_policy == best_choice[left]:
                        continue
                    for right_policy in group_policies[right]:
                        if right_policy == best_choice[right]:
                            continue
                        candidate = dict(best_choice)
                        candidate[left] = left_policy
                        candidate[right] = right_policy
                        score, rows = evaluator.evaluate(candidate)
                        pair_evaluations += 1
                        if score < pair_score:
                            pair_choice, pair_score, pair_rows = candidate, score, rows
                        if pair_evaluations >= int(max_pair_evaluations):
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
        if pair_score < best_score:
            best_choice, best_score, best_rows = pair_choice, pair_score, pair_rows
        else:
            break

    final_score, final_rows = evaluator.evaluate(best_choice)
    repeated_score, repeated_rows = evaluator.evaluate(best_choice)
    if final_score != repeated_score or [
        row["entered_experts"] for row in final_rows
    ] != [row["entered_experts"] for row in repeated_rows]:
        raise RuntimeError("exact repaired D1 replay was not repeatable")
    return best_choice, final_score, final_rows, {
        "seed_exact_crossings": int(seed_score[0]),
        "final_exact_crossings": int(final_score[0]),
        "coordinate_evaluations": coordinate_evaluations,
        "pair_evaluations": pair_evaluations,
        "total_exact_evaluations": evaluator.evaluations,
    }


def main() -> None:
    args = parse_args()
    if args.coordinate_sweeps < 0 or args.pair_passes < 0:
        raise ValueError("repair pass counts must be nonnegative")
    facts_path = args.layer_dir / "d1_layer_facts.json"
    facts = json.loads(facts_path.read_text())
    layer = int(facts["layer"])
    rates = tuple(int(value) for value in facts["rates"])
    if len(rates) != 1:
        raise ValueError("exact request repair requires one stored rate")
    rate = rates[0]
    allocation = pd.read_parquet(args.layer_dir / "d1_allocation_groups.parquet")
    experts = pd.read_parquet(args.layer_dir / "d1_allocation_experts.parquet")
    with np.load(args.layer_dir / "d1_selected_deltas.npz", allow_pickle=False) as source:
        deltas = {
            key: np.asarray(source[key])
            for key in source.files
            if key not in ("schema", "layer")
        }
    policies = _candidate_policies(deltas, rate)
    identities = _identity_rows(allocation, policies[0], rate)
    lookup = _allocation_lookup(allocation, rate)
    group_policies = {
        int(group): policies for group in identities["group"].tolist()
    }
    model_slice = load_d1_layer_slice(
        args.checkpoint, layer, device=args.device,
    )
    started = time.perf_counter()
    final_choices: dict[int, str] = {}
    final_metrics = []
    request_facts = {}
    for request_id, frame in identities.groupby("request_id", sort=True):
        capture = _load_capture(
            _capture_path(args.capture_dir, str(request_id)), layer,
        )
        request_groups = [
            {
                "group": int(row.group),
                "request_id": str(row.request_id),
                "position": int(row.position),
                "layer": layer,
            }
            for row in frame.sort_values("position", kind="stable").itertuples(index=False)
        ]
        evaluator = ExactRequestEvaluator(
            model_slice=model_slice,
            capture=capture,
            request_groups=request_groups,
            deltas=deltas,
            rate=rate,
            allocation_lookup=lookup,
        )
        choice, score, rows, search = _repair_request(
            evaluator,
            group_policies,
            coordinate_sweeps=args.coordinate_sweeps,
            pair_passes=args.pair_passes,
            max_pair_evaluations=args.max_pair_evaluations,
        )
        final_choices.update(choice)
        final_metrics.extend(rows)
        request_facts[str(request_id)] = {
            **search,
            "final_membership_pairs_changed": int(score[1]),
            "final_local_qenergy_damage": float(score[2]),
            "final_routing_mass_lost": float(score[3]),
            "final_pages": int(score[4]),
        }
        print(
            f"[exact repair] request={request_id} "
            f"crossings={search['seed_exact_crossings']}->"
            f"{search['final_exact_crossings']} "
            f"evaluations={search['total_exact_evaluations']}",
            flush=True,
        )

    choice_rows = []
    repaired_group_rows = []
    repaired_expert_rows = []
    repaired_delta = np.zeros_like(next(iter(deltas.values())), np.float32)
    for identity in identities.itertuples(index=False):
        group = int(identity.group)
        policy = final_choices[group]
        source_group = allocation[
            allocation["group"].eq(group)
            & allocation["policy"].eq(policy)
            & allocation["rate_pages_per_expert"].eq(rate)
        ]
        if len(source_group) != 1:
            raise RuntimeError("selected source allocation group is not unique")
        group_row = source_group.iloc[0].to_dict()
        group_row.update({"policy": REPAIRED, "source_policy": policy})
        repaired_group_rows.append(group_row)
        source_experts = experts[
            experts["group"].eq(group)
            & experts["policy"].eq(policy)
            & experts["rate_pages_per_expert"].eq(rate)
        ]
        if len(source_experts) != 8:
            raise RuntimeError("selected source expert allocation is incomplete")
        for row in source_experts.to_dict("records"):
            row.update({"policy": REPAIRED, "source_policy": policy})
            repaired_expert_rows.append(row)
        repaired_delta[group] = deltas[_policy_key(policy, rate)][group]
        choice_rows.append({
            "schema": REPAIR_SCHEMA,
            "layer": layer,
            "group": group,
            "request_id": str(identity.request_id),
            "position": int(identity.position),
            "rate_pages_per_expert": rate,
            "policy": REPAIRED,
            "source_policy": policy,
            "selected_group_pages": int(group_row["selected_group_pages"]),
            "local_qenergy_damage": float(group_row["local_qenergy_damage"]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(args.output_dir / "d1_exact_repair_metrics.parquet", final_metrics)
    atomic_parquet(args.output_dir / "d1_exact_repair_choices.parquet", choice_rows)
    atomic_parquet(
        args.output_dir / "d1_exact_repair_allocation_groups.parquet",
        repaired_group_rows,
    )
    atomic_parquet(
        args.output_dir / "d1_exact_repair_allocation_experts.parquet",
        repaired_expert_rows,
    )
    atomic_npz(
        args.output_dir / "d1_exact_repair_delta.npz",
        schema=np.asarray(REPAIR_SCHEMA),
        layer=np.asarray(layer, np.int64),
        rate_pages_per_expert=np.asarray(rate, np.int64),
        delta=repaired_delta,
    )
    repair_facts = {
        "schema": REPAIR_SCHEMA,
        "completed": True,
        "layer": layer,
        "next_layer": layer + 1,
        "rate_pages_per_expert": rate,
        "policy": REPAIRED,
        "candidate_policies": list(policies),
        "request_facts": request_facts,
        "wall_seconds": time.perf_counter() - started,
        "oracle_only": True,
        "runtime_input": False,
        "causal_pair_scope": "source_position_at_or_before_earliest_crossing",
        "input_hashes": {
            path.name: sha256(path)
            for path in sorted(args.layer_dir.iterdir())
            if path.is_file()
        },
        "checkpoint_index_sha256": sha256(
            args.checkpoint / "model.safetensors.index.json"
        ),
        "outputs": {},
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "d1_exact_repair_facts.json":
            repair_facts["outputs"][path.name] = {
                "sha256": sha256(path), "bytes": path.stat().st_size,
            }
    atomic_json(args.output_dir / "d1_exact_repair_facts.json", repair_facts)


if __name__ == "__main__":
    main()
