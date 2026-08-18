#!/usr/bin/env python3
"""Corrected physical-prefix wrapper for the Phase B implementation."""

from __future__ import annotations

import numpy as np
import run_phase_b_remote as implementation


def prefix_candidate_counts(
    order: np.ndarray,
    packet_bytes: int,
    layout: np.ndarray,
    logical_budget: float,
    physical: bool,
) -> tuple[int, int, int]:
    if not physical:
        count = min(len(order), int(logical_budget // packet_bytes))
        return count, count * packet_bytes, implementation.page_bytes(order[:count], packet_bytes, layout)
    pages: set[int] = set()
    inverse = np.empty_like(layout)
    inverse[layout] = np.arange(len(layout))
    count = 0
    for atom in order:
        position = int(inverse[int(atom)])
        new_pages = set(range(
            position * packet_bytes // 4096,
            (position * packet_bytes + packet_bytes - 1) // 4096 + 1,
        ))
        if len(pages | new_pages) * 4096 > logical_budget:
            break
        pages |= new_pages
        count += 1
    return count, count * packet_bytes, len(pages) * 4096


if __name__ == "__main__":
    implementation.candidate_counts = prefix_candidate_counts
    implementation.main()
