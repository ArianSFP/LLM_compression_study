"""Exact atom/page identities and progressive packet layout utilities.

The historical Phase-A selector returned only a count after it had skipped
non-fitting atoms.  This module makes the accepted IDs the source of truth:
the same IDs determine reconstruction, charged pages, and saved labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Hashable, Iterable

import numpy as np


PRECISIONS = (0, 2, 4, 8, 16)
UPGRADES = ((0, 2), (2, 4), (4, 8), (8, 16))


@dataclass(frozen=True, order=True)
class IncrementKey:
    projection: str
    atom_id: int
    before_bits: int
    after_bits: int


@dataclass
class Selection:
    atom_ids: list[int]
    pages: set[Hashable]
    logical_bytes: int


def nested_levels(atom: np.ndarray) -> dict[int, np.ndarray]:
    """Return all strict two's-complement prefixes of one INT16 master."""
    value = np.asarray(atom, dtype=np.float32)
    maximum = float(np.max(np.abs(value))) if value.size else 0.0
    scale = maximum / 32767.0 if maximum > 0 else 1.0
    master = np.clip(np.rint(value / scale), -32767, 32767).astype(np.int32)
    output = {0: np.zeros_like(value)}
    for bits in PRECISIONS[1:]:
        shift = 16 - bits
        restored = (master >> shift) << shift
        output[bits] = restored.astype(np.float32) * scale
    return output


def independent_level(atom: np.ndarray, bits: int) -> np.ndarray:
    """Independent symmetric per-atom quantizer used as a bounded control."""
    if bits not in PRECISIONS:
        raise ValueError(bits)
    value = np.asarray(atom, dtype=np.float32)
    if bits == 0:
        return np.zeros_like(value)
    qmax = max(2 ** (bits - 1) - 1, 1)
    maximum = float(np.max(np.abs(value))) if value.size else 0.0
    scale = maximum / qmax if maximum > 0 else 1.0
    code = np.clip(np.rint(value / scale), -qmax, qmax)
    return (code * scale).astype(np.float32)


def incremental_packet_bytes(length: int, before_bits: int, after_bits: int, metadata_bytes: int = 12) -> int:
    if (before_bits, after_bits) not in UPGRADES:
        raise ValueError((before_bits, after_bits))
    scale_bytes = 2 if before_bits == 0 else 0
    return math.ceil(length * (after_bits - before_bits) / 8) + scale_bytes + metadata_bytes


class PacketLayout:
    """Physical pages for independently addressable progressive increments.

    Plane streams have distinct page namespaces.  When gate/up bundling is
    enabled, either projection fetches the same combined-coordinate packet;
    this intentionally charges the unused companion payload.
    """

    def __init__(
        self,
        orders: dict[str, np.ndarray],
        lengths: dict[str, int],
        page_size: int,
        layout_id: str,
        bundle_gate_up: bool = False,
        active_counts: dict[str, int] | None = None,
    ) -> None:
        self.orders = {k: np.asarray(v, dtype=np.int64) for k, v in orders.items()}
        self.lengths = dict(lengths)
        self.page_size = int(page_size)
        self.layout_id = str(layout_id)
        self.bundle_gate_up = bool(bundle_gate_up)
        self.active_counts = {
            projection: int((active_counts or {}).get(projection, len(order)))
            for projection, order in self.orders.items()
        }
        self.inverse: dict[str, np.ndarray] = {}
        for projection, order in self.orders.items():
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order), dtype=np.int64)
            self.inverse[projection] = inverse

    def packet_bytes(self, key: IncrementKey) -> int:
        if self.bundle_gate_up and key.projection in ("gate", "up"):
            length = self.lengths["gate"] + self.lengths["up"]
        else:
            length = self.lengths[key.projection]
        return incremental_packet_bytes(length, key.before_bits, key.after_bits)

    def logical_payload_bytes(self, key: IncrementKey) -> int:
        return incremental_packet_bytes(
            self.lengths[key.projection], key.before_bits, key.after_bits
        )

    def pages_for(self, key: IncrementKey) -> set[tuple[str, str, int, int, int]]:
        projection = "gate_up" if self.bundle_gate_up and key.projection in ("gate", "up") else key.projection
        order_key = "gate" if projection == "gate_up" else key.projection
        position = int(self.inverse[order_key][key.atom_id])
        if position >= self.active_counts[order_key]:
            raise KeyError(f"atom {key.atom_id} absent from partial layout {self.layout_id}")
        packet = self.packet_bytes(key)
        start = position * packet
        stop = start + packet - 1
        plane = PRECISIONS.index(key.after_bits) - 1
        return {
            (self.layout_id, projection, plane, page, self.page_size)
            for page in range(start // self.page_size, stop // self.page_size + 1)
        }

    def serialized_bytes(self, maximum_bits: int = 16) -> int:
        total = 0
        projections = ["down"]
        if self.bundle_gate_up:
            projections.append("gate_up")
        else:
            projections.extend(["gate", "up"])
        for projection in projections:
            order_key = "gate" if projection == "gate_up" else projection
            atom_count = self.active_counts[order_key]
            for before, after in UPGRADES:
                if after > maximum_bits:
                    continue
                if projection == "gate_up":
                    length = self.lengths["gate"] + self.lengths["up"]
                else:
                    length = self.lengths[projection]
                stream = atom_count * incremental_packet_bytes(length, before, after)
                total += math.ceil(stream / self.page_size) * self.page_size
        return total


def incremental_page_cost(candidate_pages: Iterable[Hashable], selected_pages: set[Hashable], page_size: int) -> int:
    return len(set(candidate_pages) - selected_pages) * int(page_size)


def exact_id_page_selection(
    ranked_atom_ids: Iterable[int],
    page_lookup: Callable[[int], set[Hashable]],
    logical_bytes: Callable[[int], int],
    byte_budget: int,
    page_size: int,
    skip_nonfitting: bool = True,
) -> Selection:
    """Select exact accepted IDs and charge precisely their page union."""
    selected: list[int] = []
    pages: set[Hashable] = set()
    logical = 0
    for atom_id in ranked_atom_ids:
        atom = int(atom_id)
        proposed = pages | set(page_lookup(atom))
        if len(proposed) * page_size > byte_budget:
            if skip_nonfitting:
                continue
            break
        selected.append(atom)
        pages = proposed
        logical += int(logical_bytes(atom))
    return Selection(selected, pages, logical)


def reconstruct_selected(atoms: np.ndarray, codes: np.ndarray, selected_ids: Iterable[int]) -> np.ndarray:
    ids = np.asarray(list(selected_ids), dtype=np.int64)
    if not len(ids):
        return np.zeros(atoms.shape[0], dtype=np.float32)
    return np.asarray(atoms[:, ids] @ np.asarray(codes)[ids], dtype=np.float32)


def importance_layout(importance: np.ndarray) -> np.ndarray:
    return np.argsort(np.asarray(importance), kind="stable")[::-1].astype(np.int64)


def pairwise_layout(selection_incidence: np.ndarray, importance: np.ndarray | None = None) -> np.ndarray:
    selected = np.asarray(selection_incidence, dtype=np.float32)
    if selected.ndim != 2:
        raise ValueError("selection incidence must be [samples, atoms]")
    count = selected.sum(axis=0) if importance is None else np.asarray(importance, dtype=np.float32)
    affinity = selected.T @ selected
    remaining = np.ones(selected.shape[1], dtype=bool)
    first = int(np.argmax(count))
    order = [first]
    remaining[first] = False
    while remaining.any():
        score = affinity[order[-1]] + 1e-6 * count
        score[~remaining] = -np.inf
        nxt = int(np.argmax(score))
        order.append(nxt)
        remaining[nxt] = False
    return np.asarray(order, dtype=np.int64)


def hypergraph_layout(selection_incidence: np.ndarray, atoms_per_page: int, importance: np.ndarray | None = None) -> np.ndarray:
    """Greedily fill pages using total within-page training co-selection."""
    selected = np.asarray(selection_incidence, dtype=np.float32)
    count = selected.sum(axis=0) if importance is None else np.asarray(importance, dtype=np.float32)
    affinity = selected.T @ selected
    remaining = np.ones(selected.shape[1], dtype=bool)
    order: list[int] = []
    capacity = max(int(atoms_per_page), 1)
    while remaining.any():
        seed_score = count.copy()
        seed_score[~remaining] = -np.inf
        seed = int(np.argmax(seed_score))
        page = [seed]
        remaining[seed] = False
        while len(page) < capacity and remaining.any():
            score = affinity[:, page].sum(axis=1) + 1e-4 * count
            score[~remaining] = -np.inf
            nxt = int(np.argmax(score))
            page.append(nxt)
            remaining[nxt] = False
        order.extend(page)
    return np.asarray(order, dtype=np.int64)


def regime_layouts(selection_incidence: np.ndarray, replicas: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Training-only k-means regimes over binary support masks."""
    incidence = np.asarray(selection_incidence, dtype=np.float32)
    if len(incidence) == 0:
        raise ValueError("cannot fit a regime layout without training selections")
    if replicas <= 1:
        return [importance_layout(incidence.mean(axis=0))], np.zeros(len(incidence), dtype=np.int64)
    effective_replicas = min(int(replicas), len(incidence))
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((incidence.shape[1], min(16, incidence.shape[1])), dtype=np.float32)
    features = incidence @ projection / math.sqrt(projection.shape[1])
    centres = features[rng.choice(len(features), size=effective_replicas, replace=False)].copy()
    assignment = np.zeros(len(features), dtype=np.int64)
    for _ in range(20):
        distance = ((features[:, None] - centres[None]) ** 2).sum(axis=2)
        new_assignment = np.argmin(distance, axis=1)
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for cluster in range(effective_replicas):
            members = features[assignment == cluster]
            if len(members):
                centres[cluster] = members.mean(axis=0)
    layouts = []
    for cluster in range(effective_replicas):
        mask = assignment == cluster
        importance = incidence[mask].mean(axis=0) if mask.any() else incidence.mean(axis=0)
        layouts.append(importance_layout(importance))
    # A cold expert may not have enough distinct training masks to identify
    # every requested physical replica. Preserve the requested storage count
    # with deterministic duplicate layouts instead of leaking validation/test
    # masks or failing the experiment.
    while len(layouts) < replicas:
        layouts.append(layouts[len(layouts) % effective_replicas].copy())
    return layouts, assignment


def storage_multiplier(reference_bytes: int, layouts: list[PacketLayout], maximum_bits: list[int], resident_basis_bytes: int = 0) -> dict[str, float | int]:
    external = int(reference_bytes + sum(layout.serialized_bytes(bits) for layout, bits in zip(layouts, maximum_bits)))
    return {
        "reference_bytes": int(reference_bytes),
        "external_bytes": external,
        "external_multiplier": float(external / reference_bytes),
        "resident_basis_bytes": int(resident_basis_bytes),
    }
