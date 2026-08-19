"""Page-accounted actions for bidirectional and parity MXFP4 descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
from typing import Iterable, Mapping

import numpy as np

from .mxfp4_descriptions import DESCRIPTION_MASK, DESCRIPTIONS


@dataclass(frozen=True)
class DescriptionAction:
    coordinate: int
    before_mask: int
    add_descriptions: tuple[str, ...]
    after_mask: int
    score: float


def mask_descriptions(mask: int) -> tuple[str, ...]:
    return tuple(name for name in DESCRIPTIONS if mask & DESCRIPTION_MASK[name])


def is_exact_mask(mask: int) -> bool:
    return int(mask).bit_count() >= 2


def description_to_page(
    layout: np.ndarray,
    coordinate: int,
    description: str,
    payload_bytes: int,
    page_size: int,
    available_descriptions: Iterable[str],
) -> int:
    """Map an exact description packet to its physical plane page."""
    names = tuple(available_descriptions)
    if description not in names:
        raise ValueError(description)
    layout = np.asarray(layout, dtype=np.int64)
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    per_page = page_size // payload_bytes
    if per_page < 1:
        raise ValueError("one description packet does not fit in a page")
    pages_per_plane = (len(layout) + per_page - 1) // per_page
    return names.index(description) * pages_per_plane + int(inverse[coordinate]) // per_page


def transition_score(target: np.ndarray, before: np.ndarray, after: np.ndarray) -> float:
    """Exact within-coordinate squared-error reduction for one transition."""
    previous = target.astype(np.float64) - before.astype(np.float64)
    current = target.astype(np.float64) - after.astype(np.float64)
    return float(previous @ previous - current @ current)


def _initial_actions(
    target: np.ndarray,
    single: Mapping[str, np.ndarray],
    descriptions: tuple[str, ...],
    first_descriptions: tuple[str, ...],
    allow_direct_pairs: bool,
) -> list[DescriptionAction]:
    coordinates = target.shape[0]
    actions: list[DescriptionAction] = []
    zero = np.zeros(target.shape[1], dtype=np.float32)
    for coordinate in range(coordinates):
        local_target = target[coordinate]
        for name in first_descriptions:
            score = transition_score(local_target, zero, single[name][coordinate])
            actions.append(DescriptionAction(
                coordinate, 0, (name,), DESCRIPTION_MASK[name], score,
            ))
        for left, right in itertools.combinations(descriptions, 2) if allow_direct_pairs else ():
            # Any two distinct descriptions give the exact local contribution.
            pair_mask = DESCRIPTION_MASK[left] | DESCRIPTION_MASK[right]
            score = transition_score(local_target, zero, local_target)
            actions.append(DescriptionAction(
                coordinate, 0, (left, right), pair_mask, score,
            ))
    return actions


def select_description_actions(
    target_contributions: np.ndarray,
    single_contributions: Mapping[str, np.ndarray],
    layout: np.ndarray,
    payload_bytes: int,
    page_size: int,
    byte_budget: int,
    available_descriptions: Iterable[str],
    first_descriptions: Iterable[str] | None = None,
    allow_direct_pairs: bool = True,
) -> tuple[list[DescriptionAction], set[int], np.ndarray]:
    """Static within-coordinate-marginal oracle with exact page/state accounting.

    Candidate priorities are damage reduction per logical one-bit packet.  Page
    admission uses the exact union of physical description-plane pages.  The
    method deliberately remains a scalable diagonal oracle; interaction-aware
    search is reported separately rather than being conflated with this codec
    comparison.
    """
    target = np.asarray(target_contributions, dtype=np.float32)
    descriptions = tuple(available_descriptions)
    if descriptions not in (("a", "b"), ("a", "b", "q")):
        raise ValueError("descriptions must be ordered as (a,b) or (a,b,q)")
    first = descriptions if first_descriptions is None else tuple(first_descriptions)
    if not first or any(name not in descriptions for name in first):
        raise ValueError("first descriptions must be a non-empty subset of available descriptions")
    single = {name: np.asarray(single_contributions[name], dtype=np.float32) for name in descriptions}
    if target.ndim != 2 or any(value.shape != target.shape for value in single.values()):
        raise ValueError("target and single-description contributions must share [coordinate, output] shape")
    heap: list[tuple[float, int, DescriptionAction]] = []
    serial = 0

    def push(action: DescriptionAction) -> None:
        nonlocal serial
        # Negative-gain actions are never useful before considering cross-coordinate
        # interactions, which this explicitly diagonal allocator does not model.
        priority = action.score / len(action.add_descriptions)
        heapq.heappush(heap, (-priority, serial, action))
        serial += 1

    for action in _initial_actions(target, single, descriptions, first, allow_direct_pairs):
        push(action)
    states = np.zeros(target.shape[0], dtype=np.uint8)
    pages: set[int] = set()
    selected: list[DescriptionAction] = []
    while heap:
        negative, _, action = heapq.heappop(heap)
        if action.score <= 0 or int(states[action.coordinate]) != action.before_mask:
            continue
        action_pages = {
            description_to_page(layout, action.coordinate, name, payload_bytes, page_size, descriptions)
            for name in action.add_descriptions
        }
        proposed = pages | action_pages
        if len(proposed) * page_size > byte_budget:
            continue
        pages = proposed
        states[action.coordinate] = action.after_mask
        selected.append(action)
        if not is_exact_mask(action.after_mask):
            first = mask_descriptions(action.after_mask)[0]
            remaining = target[action.coordinate] - single[first][action.coordinate]
            score = float(remaining.astype(np.float64) @ remaining.astype(np.float64))
            for name in descriptions:
                if name == first:
                    continue
                push(DescriptionAction(
                    action.coordinate,
                    action.after_mask,
                    (name,),
                    action.after_mask | DESCRIPTION_MASK[name],
                    score,
                ))
    return selected, pages, states


def reconstruct_selected_correction(
    target_contributions: np.ndarray,
    single_contributions: Mapping[str, np.ndarray],
    states: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_contributions, dtype=np.float32)
    state = np.asarray(states, dtype=np.uint8)
    if state.shape != (target.shape[0],):
        raise ValueError("state geometry mismatch")
    output = np.zeros(target.shape[1], dtype=np.float32)
    for coordinate, mask in enumerate(state):
        if is_exact_mask(int(mask)):
            output += target[coordinate]
        elif mask:
            name = mask_descriptions(int(mask))[0]
            output += np.asarray(single_contributions[name], dtype=np.float32)[coordinate]
    return output


def action_page_ids(
    actions: Iterable[DescriptionAction],
    layout: np.ndarray,
    payload_bytes: int,
    page_size: int,
    available_descriptions: Iterable[str],
) -> list[list[int]]:
    descriptions = tuple(available_descriptions)
    return [[
        description_to_page(layout, action.coordinate, name, payload_bytes, page_size, descriptions)
        for name in action.add_descriptions
    ] for action in actions]
