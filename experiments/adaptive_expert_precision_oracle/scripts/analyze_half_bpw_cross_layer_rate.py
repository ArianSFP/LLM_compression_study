#!/usr/bin/env python3
"""Merge compact cross-layer probes and compare rate-allocation objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


GROUP_PAGES_AT_HALF_BPW = 8 * 384


def _objective(policy: str, mode: str) -> Callable[[Mapping[str, Any]], float]:
    field = (
        "predicted_normalized_damage" if policy == "compressed"
        else "exact_normalized_damage"
    )
    if mode == "sum":
        return lambda row: float(row[field])
    if mode == "square":
        return lambda row: float(row[field]) ** 2
    if mode == "fourth":
        return lambda row: float(row[field]) ** 4
    if mode == "hinge99":
        return lambda row: max(float(row[field]) - 0.01, 0.0)
    if mode == "hinge995":
        return lambda row: max(float(row[field]) - 0.005, 0.0)
    raise ValueError(mode)


def _allocate(
    groups: Sequence[Mapping[str, Any]], policy: str, page_budget: int,
    mode: str,
) -> tuple[list[int], int]:
    score = _objective(policy, mode)
    dynamic = np.full(int(page_budget) + 1, np.inf, np.float64)
    dynamic[0] = 0.0
    choices = np.full((len(groups), int(page_budget) + 1), -1, np.int16)
    previous = np.full((len(groups), int(page_budget) + 1), -1, np.int32)
    for group_index, group in enumerate(groups):
        updated = np.full_like(dynamic, np.inf)
        finite = np.flatnonzero(np.isfinite(dynamic))
        for option_index, option in enumerate(group[policy]):
            pages = int(option["pages"])
            source = finite[finite + pages <= int(page_budget)]
            target = source + pages
            candidate = dynamic[source] + score(option)
            better = candidate < updated[target] - 1e-15
            ties = np.abs(candidate - updated[target]) <= 1e-15
            better |= ties & (
                (choices[group_index, target] < 0)
                | (option_index < choices[group_index, target])
            )
            selected = np.flatnonzero(better)
            if selected.size:
                positions = target[selected]
                updated[positions] = candidate[selected]
                choices[group_index, positions] = int(option_index)
                previous[group_index, positions] = source[selected]
        dynamic = updated
    used = min(
        np.flatnonzero(np.isfinite(dynamic)).tolist(),
        key=lambda pages: (float(dynamic[pages]), pages),
    )
    selected = [0] * len(groups)
    cursor = int(used)
    for group_index in range(len(groups) - 1, -1, -1):
        option = int(choices[group_index, cursor])
        if option < 0:
            raise RuntimeError("allocation backtrack failed")
        selected[group_index] = option
        cursor = int(previous[group_index, cursor])
    return selected, int(used)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _merge_options(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    objective = (
        "predicted_normalized_damage" if policy == "compressed"
        else "exact_normalized_damage"
    )
    by_pages: dict[int, dict[str, Any]] = {}
    for raw in (*first, *second):
        row = dict(raw)
        pages = int(row["pages"])
        old = by_pages.get(pages)
        if old is None or float(row[objective]) < float(old[objective]) - 1e-15:
            by_pages[pages] = row
    result, best = [], np.inf
    for pages in sorted(by_pages):
        row = by_pages[pages]
        if float(row[objective]) < best - 1e-15:
            result.append(row)
            best = float(row[objective])
    return result


def _tail_floor_allocate(
    groups: Sequence[Mapping[str, Any]], policy: str, page_budget: int,
    quantile: float = 0.1,
) -> tuple[list[int], int, float, int, list[int], list[dict[str, Any]]]:
    keep = len(groups) - int(np.floor(float(quantile) * (len(groups) - 1)))
    selector_score = (
        (lambda row: 1.0 - float(row["predicted_normalized_damage"]))
        if policy == "compressed" else (lambda row: float(row["recovery"]))
    )

    def costs(threshold: float) -> list[tuple[int, int, int]]:
        result = []
        for group_index, group in enumerate(groups):
            feasible = [
                (int(option["pages"]), option_index)
                for option_index, option in enumerate(group[policy])
                if selector_score(option) >= float(threshold)
            ]
            result.append((
                *(min(feasible) if feasible else (10**9, -1)), group_index,
            ))
        return result

    low, high = 0.0, 1.0
    for _ in range(55):
        middle = 0.5 * (low + high)
        required = sum(item[0] for item in sorted(costs(middle))[:keep])
        if required <= int(page_budget):
            low = middle
        else:
            high = middle
    protected = {item[2] for item in sorted(costs(low - 1e-12))[:keep]}
    filtered = []
    for group_index, group in enumerate(groups):
        record = dict(group)
        record[policy] = [
            option for option in group[policy]
            if group_index not in protected or selector_score(option) >= low - 1e-12
        ]
        filtered.append(record)
    selected, used = _allocate(filtered, policy, page_budget, "fourth")
    target_pages = sum(item[0] for item in sorted(costs(0.99))[:keep])
    unprotected = sorted(set(range(len(groups))) - protected)
    return selected, used, float(low), int(target_pages), unprotected, filtered


def _baseline_option(group: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    field = (
        "predicted_normalized_damage" if policy == "compressed"
        else "exact_normalized_damage"
    )
    feasible = [
        option for option in group[policy]
        if int(option["pages"]) <= GROUP_PAGES_AT_HALF_BPW
    ]
    return min(feasible, key=lambda row: (float(row[field]), int(row["pages"])))


def _mean_from_source(source: str) -> int:
    return int(str(source).rsplit("_", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--supplements", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in args.inputs:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "half_bpw_cross_layer_compact_probe_v1":
            raise RuntimeError(f"unexpected probe schema: {path}")
        for raw in payload["groups"]:
            identity = (
                str(raw["request_id"]), int(raw["position"]), int(raw["layer"]),
            )
            if identity in grouped:
                raise RuntimeError("duplicate group identities across base probe chunks")
            grouped[identity] = dict(raw)
    for path in args.supplements:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "half_bpw_cross_layer_compact_probe_v1":
            raise RuntimeError(f"unexpected supplemental probe schema: {path}")
        for raw in payload["groups"]:
            identity = (
                str(raw["request_id"]), int(raw["position"]), int(raw["layer"]),
            )
            if identity not in grouped:
                raise RuntimeError("supplement contains an identity outside the base grid")
            for policy in ("compressed", "exact"):
                grouped[identity][policy] = _merge_options(
                    grouped[identity][policy], raw[policy], policy,
                )
    groups = list(grouped.values())
    by_token: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for group in groups:
        by_token.setdefault(
            (str(group["request_id"]), int(group["position"])), [],
        ).append(group)
    layer_count = len({int(group["layer"]) for group in groups})
    if any(len(rows) != layer_count for rows in by_token.values()):
        raise RuntimeError("token/layer grid is incomplete")
    for rows in by_token.values():
        rows.sort(key=lambda row: int(row["layer"]))

    result: dict[str, Any] = {
        "schema": "half_bpw_cross_layer_compact_analysis_v1",
        "groups": len(groups), "layers": layer_count, "tokens": len(by_token),
        "allowed_pages": len(groups) * GROUP_PAGES_AT_HALF_BPW,
        "policies": {},
    }
    for policy in ("compressed", "exact"):
        baseline = [
            float(_baseline_option(group, policy)["recovery"])
            for group in groups
        ]
        policy_result: dict[str, Any] = {"uniform_0p5": _summarize(baseline)}
        for mode in ("sum", "square", "fourth", "hinge99", "hinge995"):
            recoveries, used_pages = [], 0
            selections: dict[tuple[str, int], list[int]] = {}
            for token, token_groups in by_token.items():
                selected, used = _allocate(
                    token_groups, policy,
                    len(token_groups) * GROUP_PAGES_AT_HALF_BPW, mode,
                )
                selections[token] = selected
                used_pages += used
                recoveries.extend(
                    float(group[policy][choice]["recovery"])
                    for group, choice in zip(token_groups, selected)
                )
            policy_result[f"pooled_{mode}"] = {
                **_summarize(recoveries),
                "used_pages": int(used_pages),
                "paired_delta": _summarize(
                    np.asarray(recoveries) - np.asarray(baseline)
                ),
            }

            heldout = []
            tokens = sorted(by_token)
            if len(tokens) >= 2 and not args.supplements:
                for train_token in tokens:
                    schedule = {
                        int(group["layer"]): _mean_from_source(
                            group[policy][choice]["source"]
                        )
                        for group, choice in zip(
                            by_token[train_token], selections[train_token]
                        )
                    }
                    for test_token in tokens:
                        if test_token == train_token:
                            continue
                        for group in by_token[test_token]:
                            mean = schedule[int(group["layer"])]
                            matches = [
                                option for option in group[policy]
                                if _mean_from_source(option["source"]) == mean
                            ]
                            if len(matches) != 1:
                                raise RuntimeError("static schedule option missing")
                            heldout.append(float(matches[0]["recovery"]))
                policy_result[f"static_leave_one_token_out_{mode}"] = (
                    _summarize(heldout)
                )
        tail_recoveries, tail_pages, token_records = [], 0, []
        for token, token_groups in sorted(by_token.items()):
            budget = len(token_groups) * GROUP_PAGES_AT_HALF_BPW
            (
                selected, used, floor, target_pages, unprotected, filtered,
            ) = _tail_floor_allocate(token_groups, policy, budget)
            recoveries = [
                float(group[policy][choice]["recovery"])
                for group, choice in zip(filtered, selected)
            ]
            tail_recoveries.extend(recoveries)
            tail_pages += used
            token_records.append({
                "request_id": token[0], "position": int(token[1]),
                "selector_floor": floor, "observed": _summarize(recoveries),
                "used_pages": int(used),
                "minimum_pages_for_37_of_40_at_99": int(target_pages),
                "target_99_feasible": int(target_pages) <= budget,
                "unprotected_layers": [
                    int(token_groups[index]["layer"]) for index in unprotected
                ],
            })
        policy_result["tail_floor_37_of_40"] = {
            **_summarize(tail_recoveries), "used_pages": int(tail_pages),
            "selection_statistic": (
                "predicted_normalized_damage" if policy == "compressed"
                else "exact_validation_recovery_oracle"
            ),
            "tokens": token_records,
        }
        result["policies"][policy] = policy_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
