"""Exact nested-bit selection and page accounting for embedded MXFP4."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BitAction:
    coordinate: int
    stage: int
    score: float


def diagonal_nested_order(stage_scores: Iterable[np.ndarray]) -> list[BitAction]:
    """Order nested Q2->Q3->Q4 actions using static diagonal scores."""
    scores = [np.asarray(value, dtype=np.float64) for value in stage_scores]
    if len(scores) != 2 or scores[0].shape != scores[1].shape:
        raise ValueError("two equal-shaped refinement score vectors are required")
    heap: list[tuple[float, int, int]] = []
    depth = np.zeros(scores[0].shape, dtype=np.uint8)
    for coordinate, score in enumerate(scores[0]):
        heapq.heappush(heap, (-float(score), coordinate, 1))
    order: list[BitAction] = []
    while heap:
        negative, coordinate, stage = heapq.heappop(heap)
        if stage != int(depth[coordinate]) + 1:
            raise AssertionError("non-nested action queue")
        depth[coordinate] = stage
        order.append(BitAction(coordinate, stage, -negative))
        if stage == 1:
            heapq.heappush(heap, (-float(scores[1][coordinate]), coordinate, 2))
    return order


def coordinate_to_page(layout: np.ndarray, coordinate: int, stage: int, payload_bytes: int, page_size: int) -> int:
    layout = np.asarray(layout, dtype=np.int64)
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    per_page = page_size // payload_bytes
    if per_page < 1:
        raise ValueError("one direction does not fit in a page")
    pages_per_plane = (len(layout) + per_page - 1) // per_page
    return (stage - 1) * pages_per_plane + int(inverse[coordinate]) // per_page


def pairwise_coselection_layout(incidence: np.ndarray, group_size: int) -> np.ndarray:
    """Greedy training-only page groups based on pairwise mask incidence."""
    values = np.asarray(incidence, dtype=np.uint8)
    if values.ndim != 2 or group_size < 1:
        raise ValueError("invalid incidence matrix or group size")
    remaining = np.ones(values.shape[1], dtype=bool)
    frequency = values.sum(axis=0).astype(np.int64)
    result: list[int] = []
    while np.any(remaining):
        eligible = np.flatnonzero(remaining)
        seed = int(eligible[np.argmax(frequency[eligible])])
        group = [seed]
        remaining[seed] = False
        while len(group) < group_size and np.any(remaining):
            eligible = np.flatnonzero(remaining)
            similarity = values[:, eligible].T.astype(np.int64) @ values[:, group].sum(axis=1).astype(np.int64)
            chosen = int(eligible[np.lexsort((frequency[eligible], similarity))[-1]])
            group.append(chosen)
            remaining[chosen] = False
        result.extend(group)
    return np.asarray(result, dtype=np.int64)


def select_under_page_budget(
    order: Iterable[BitAction], layout: np.ndarray, payload_bytes: int, page_size: int,
    byte_budget: int,
) -> tuple[list[BitAction], set[int]]:
    selected: list[BitAction] = []
    pages: set[int] = set()
    depth = np.zeros(len(layout), dtype=np.uint8)
    for action in order:
        if action.stage != int(depth[action.coordinate]) + 1:
            continue
        page = coordinate_to_page(layout, action.coordinate, action.stage, payload_bytes, page_size)
        new_pages = pages | {page}
        if len(new_pages) * page_size > byte_budget:
            continue
        pages = new_pages
        depth[action.coordinate] = action.stage
        selected.append(action)
    return selected, pages


def exact_marginal_order(contributions: np.ndarray) -> list[BitAction]:
    """Exact projection-residual OMP ordering with nested stage eligibility."""
    values = np.asarray(contributions, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("expected [2, coordinates, output] contributions")
    stages, coordinates, output = values.shape
    flat = values.reshape(stages * coordinates, output)
    gram = flat @ flat.T
    correlation = gram.sum(axis=1)
    norms = np.diag(gram)
    selected = np.zeros(len(flat), dtype=bool)
    depth = np.zeros(coordinates, dtype=np.uint8)
    result: list[BitAction] = []
    for _ in range(len(flat)):
        eligible = np.zeros(len(flat), dtype=bool)
        eligible[:coordinates] = depth == 0
        eligible[coordinates:] = depth == 1
        marginal = 2.0 * correlation - norms
        marginal[~eligible | selected] = -np.inf
        chosen = int(np.argmax(marginal))
        if not np.isfinite(marginal[chosen]):
            break
        stage = chosen // coordinates + 1
        coordinate = chosen % coordinates
        selected[chosen] = True
        depth[coordinate] = stage
        result.append(BitAction(coordinate, stage, float(marginal[chosen])))
        correlation -= gram[:, chosen]
    return result


def backward_elimination_order(contributions: np.ndarray) -> list[BitAction]:
    """Return a forward order induced by exact nested backward elimination."""
    values = np.asarray(contributions, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("expected [2, coordinates, output] contributions")
    stages, coordinates, output = values.shape
    flat = values.reshape(stages * coordinates, output)
    gram = flat @ flat.T
    residual_correlation = np.zeros(len(flat), dtype=np.float64)
    norms = np.diag(gram)
    present = np.ones(len(flat), dtype=bool)
    depth = np.full(coordinates, 2, dtype=np.uint8)
    removed: list[BitAction] = []
    for _ in range(len(flat)):
        eligible = np.zeros(len(flat), dtype=bool)
        eligible[coordinates:] = depth == 2
        eligible[:coordinates] = depth == 1
        increase = 2.0 * residual_correlation + norms
        increase[~eligible | ~present] = np.inf
        chosen = int(np.argmin(increase))
        if not np.isfinite(increase[chosen]):
            break
        stage = chosen // coordinates + 1
        coordinate = chosen % coordinates
        present[chosen] = False
        depth[coordinate] = stage - 1
        removed.append(BitAction(coordinate, stage, float(increase[chosen])))
        residual_correlation += gram[:, chosen]
    return list(reversed(removed))
