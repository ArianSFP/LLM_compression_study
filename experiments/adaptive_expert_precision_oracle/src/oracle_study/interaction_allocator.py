"""Residual-aware fixed-coefficient selectors for embedded MXFP4 actions.

The locked Q2->Q3->Q4 codec supplies two delta matrices per projection. This
module changes only allocation: deltas, coefficients, trees, and the endpoint
are never refitted. Actions are flattened stage-major.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np

from .mxfp4_selective import BitAction, diagonal_nested_order


CANONICAL_EXACT_LABEL = "exact_marginal_fixed_greedy"


def _flat_deltas(deltas: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(deltas, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("expected deltas with shape [2, coordinates, output]")
    return values.reshape(2 * values.shape[1], values.shape[2]), int(values.shape[1])


def expanded_activation(activation: np.ndarray, coordinates: int) -> np.ndarray:
    values = np.asarray(activation, dtype=np.float32).reshape(-1)
    if values.size == coordinates:
        return np.concatenate((values, values))
    if values.size == 2 * coordinates:
        return values.copy()
    raise ValueError(f"activation has {values.size} values; expected {coordinates} or {2 * coordinates}")


def build_base_gram(
    deltas: np.ndarray,
    metric: np.ndarray | None = None,
    *,
    device: str = "cuda",
) -> np.ndarray:
    """Build ``K = D.T @ G @ D`` once, preferably on the GPU.

    ``deltas`` is ``[stage, coordinate, output]``; its flattened rows are
    ``D.T``. The returned CPU float32 matrix is reusable across activations.
    """
    flat, _ = _flat_deltas(deltas)
    if device == "cpu":
        if metric is None:
            return np.asarray(flat @ flat.T, dtype=np.float32)
        g = np.asarray(metric, dtype=np.float32)
        if g.shape != (flat.shape[1], flat.shape[1]):
            raise ValueError("metric shape does not match projection output")
        return np.asarray((flat @ g) @ flat.T, dtype=np.float32)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        values = torch.from_numpy(flat).to(device)
        if metric is None:
            gram = values @ values.T
        else:
            g = torch.as_tensor(np.asarray(metric, np.float32), device=device)
            if tuple(g.shape) != (flat.shape[1], flat.shape[1]):
                raise ValueError("metric shape does not match projection output")
            gram = (values @ g) @ values.T
        return gram.cpu().numpy().astype(np.float32, copy=False)
    except (ImportError, RuntimeError):
        if device != "cuda":
            raise
        return build_base_gram(deltas, metric, device="cpu")


def direct_action_gram(contributions: np.ndarray, metric: np.ndarray | None = None) -> np.ndarray:
    """Reference action Gram used only for equivalence tests and audits."""
    flat, _ = _flat_deltas(contributions)
    if metric is None:
        return np.asarray(flat @ flat.T, np.float32)
    g = np.asarray(metric, np.float32)
    return np.asarray((flat @ g) @ flat.T, np.float32)


def _eligible(depth: np.ndarray) -> np.ndarray:
    coordinates = len(depth)
    result = np.zeros(2 * coordinates, dtype=bool)
    result[:coordinates] = depth == 0
    result[coordinates:] = depth == 1
    return result


def _action(index: int, coordinates: int, score: float) -> BitAction:
    return BitAction(index % coordinates, index // coordinates + 1, float(score))


def exact_marginal_fixed_greedy_from_gram(
    gram: np.ndarray, activation: np.ndarray, coordinates: int | None = None,
) -> list[BitAction]:
    """Exact marginal path with fixed coefficients and nested eligibility.

    The score is ``2*a_j*(K@v)_j - a_j**2*K_jj``. Selecting ``j`` sets
    ``v_j=0`` and updates the correlation from one precomputed Gram column.
    """
    k = np.asarray(gram, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] % 2:
        raise ValueError("gram must be square with two equal action planes")
    n = k.shape[0] // 2 if coordinates is None else int(coordinates)
    if k.shape != (2 * n, 2 * n):
        raise ValueError("coordinates do not match gram")
    a = expanded_activation(activation, n)
    correlation = k @ a
    diagonal = np.diag(k).copy()
    depth = np.zeros(n, dtype=np.uint8)
    selected = np.zeros(2 * n, dtype=bool)
    result: list[BitAction] = []
    for _ in range(2 * n):
        marginal = 2.0 * a * correlation - a * a * diagonal
        marginal[~_eligible(depth) | selected] = -np.inf
        chosen = int(np.argmax(marginal))
        if not np.isfinite(marginal[chosen]):
            break
        result.append(_action(chosen, n, float(marginal[chosen])))
        selected[chosen] = True
        depth[chosen % n] = chosen // n + 1
        correlation -= float(a[chosen]) * k[:, chosen]
    return result


def diagonal_from_gram(gram: np.ndarray, activation: np.ndarray) -> list[BitAction]:
    k = np.asarray(gram, dtype=np.float32)
    n = k.shape[0] // 2
    a = expanded_activation(activation, n)
    scores = a * a * np.diag(k)
    return diagonal_nested_order((scores[:n], scores[n:]))


def initial_full_marginal_from_gram(gram: np.ndarray, activation: np.ndarray) -> list[BitAction]:
    k = np.asarray(gram, dtype=np.float32)
    n = k.shape[0] // 2
    a = expanded_activation(activation, n)
    scores = 2.0 * a * (k @ a) - a * a * np.diag(k)
    return diagonal_nested_order((scores[:n], scores[n:]))


def block_refresh_from_gram(
    gram: np.ndarray,
    activation: np.ndarray,
    block_size: int,
    *,
    shortlist_size: int = 128,
) -> list[BitAction]:
    """Refresh globally, then choose up to ``block_size`` from a shortlist."""
    if block_size < 1 or shortlist_size < 1:
        raise ValueError("block and shortlist sizes must be positive")
    k = np.asarray(gram, dtype=np.float32)
    n = k.shape[0] // 2
    a = expanded_activation(activation, n)
    correlation = k @ a
    diagonal = np.diag(k).copy()
    depth = np.zeros(n, dtype=np.uint8)
    selected = np.zeros(2 * n, dtype=bool)
    result: list[BitAction] = []
    while len(result) < 2 * n:
        global_score = 2.0 * a * correlation - a * a * diagonal
        ids = np.flatnonzero(_eligible(depth) & ~selected)
        if not len(ids):
            break
        ranked = ids[np.lexsort((ids, -global_score[ids]))]
        shortlist = ranked[: min(shortlist_size, len(ranked))]
        shortlist_live = np.ones(len(shortlist), dtype=bool)
        selected_this_refresh = 0
        while selected_this_refresh < block_size and np.any(shortlist_live):
            scores = 2.0 * a[shortlist] * correlation[shortlist] - a[shortlist] ** 2 * diagonal[shortlist]
            valid = shortlist_live & _eligible(depth)[shortlist] & ~selected[shortlist]
            scores[~valid] = -np.inf
            position = int(np.argmax(scores))
            if not np.isfinite(scores[position]):
                break
            chosen = int(shortlist[position])
            result.append(_action(chosen, n, float(scores[position])))
            selected[chosen] = True
            depth[chosen % n] = chosen // n + 1
            correlation -= float(a[chosen]) * k[:, chosen]
            shortlist_live[position] = False
            selected_this_refresh += 1
    return result


def backward_elimination_from_gram(gram: np.ndarray, activation: np.ndarray) -> list[BitAction]:
    k = np.asarray(gram, dtype=np.float32)
    n = k.shape[0] // 2
    a = expanded_activation(activation, n)
    action_diagonal = a * a * np.diag(k)
    residual_correlation = np.zeros(2 * n, dtype=np.float32)
    present = np.ones(2 * n, dtype=bool)
    depth = np.full(n, 2, dtype=np.uint8)
    removed: list[BitAction] = []
    for _ in range(2 * n):
        eligible = np.zeros(2 * n, dtype=bool)
        eligible[n:] = depth == 2
        eligible[:n] = depth == 1
        increase = 2.0 * a * residual_correlation + action_diagonal
        increase[~eligible | ~present] = np.inf
        chosen = int(np.argmin(increase))
        if not np.isfinite(increase[chosen]):
            break
        removed.append(_action(chosen, n, float(increase[chosen])))
        present[chosen] = False
        depth[chosen % n] = chosen // n
        residual_correlation += a * k[:, chosen] * float(a[chosen])
    return list(reversed(removed))


def forward_backward_hybrid_from_gram(
    gram: np.ndarray, activation: np.ndarray, forward_actions: int,
) -> list[BitAction]:
    """Use an exact forward prefix and a nested backward-elimination tail."""
    forward = exact_marginal_fixed_greedy_from_gram(gram, activation)
    backward = backward_elimination_from_gram(gram, activation)
    limit = min(max(int(forward_actions), 0), len(forward))
    result = list(forward[:limit])
    n = len(forward) // 2
    depth = np.zeros(n, dtype=np.uint8)
    used: set[tuple[int, int]] = set()
    for action in result:
        depth[action.coordinate] = action.stage
        used.add((action.coordinate, action.stage))
    pending = [a for a in backward if (a.coordinate, a.stage) not in used]
    while pending:
        next_pending = []
        progressed = False
        for action in pending:
            if action.stage == int(depth[action.coordinate]) + 1:
                result.append(action)
                depth[action.coordinate] = action.stage
                progressed = True
            else:
                next_pending.append(action)
        if not progressed:
            raise AssertionError("hybrid tail could not satisfy nested eligibility")
        pending = next_pending
    return result


def action_ids(order: Iterable[BitAction], coordinates: int) -> np.ndarray:
    return np.asarray([(a.stage - 1) * coordinates + a.coordinate for a in order], dtype=np.int64)


def recovery_curve_from_gram(
    order: Sequence[BitAction], gram: np.ndarray, activation: np.ndarray,
) -> np.ndarray:
    k = np.asarray(gram, dtype=np.float64)
    n = k.shape[0] // 2
    a = expanded_activation(activation, n).astype(np.float64)
    correlation = k @ a
    denominator = max(float(a @ correlation), 1e-30)
    recovered = 0.0
    curve = []
    for action in order:
        j = (action.stage - 1) * n + action.coordinate
        gain = 2.0 * a[j] * correlation[j] - a[j] * a[j] * k[j, j]
        recovered += float(gain)
        correlation -= a[j] * k[:, j]
        curve.append(recovered / denominator)
    if len(curve) == 2 * n:
        curve[-1] = 1.0
    return np.asarray(curve, dtype=np.float64)


def support_overlap(left: Sequence[BitAction], right: Sequence[BitAction], count: int) -> float:
    if count <= 0:
        return float("nan")
    left_keys = {(a.coordinate, a.stage) for a in left[:count]}
    right_keys = {(a.coordinate, a.stage) for a in right[:count]}
    return len(left_keys & right_keys) / max(len(left_keys | right_keys), 1)


def metric_features(deltas: np.ndarray, metric: np.ndarray | None = None) -> np.ndarray:
    """Return action features whose row Gram equals the requested metric Gram."""
    flat, _ = _flat_deltas(deltas)
    if metric is None:
        return flat
    g = np.asarray(metric, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh((g + g.T) * 0.5)
    keep = eigenvalues > max(float(eigenvalues.max()), 1.0) * 1e-12
    root = eigenvectors[:, keep] * np.sqrt(np.maximum(eigenvalues[keep], 0.0))[None, :]
    return np.asarray(flat.astype(np.float64) @ root, dtype=np.float32)


def low_rank_factor(
    features: np.ndarray,
    rank: int,
    method: str,
    *,
    training_activation: np.ndarray | None = None,
    seed: int = 20260818,
) -> np.ndarray:
    """Construct truncated-SVD, fixed-JL, or activation-weighted factors."""
    values = np.asarray(features, dtype=np.float32)
    r = min(int(rank), min(values.shape))
    if r < 1:
        raise ValueError("rank must be positive")
    if method == "fixed_jl":
        rng = np.random.default_rng(seed)
        sketch = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=(values.shape[1], r)) / np.sqrt(r)
        return np.asarray(values @ sketch, np.float32)
    if method == "truncated_svd":
        _, _, vt = np.linalg.svd(values, full_matrices=False)
        return np.asarray(values @ vt[:r].T, np.float32)
    if method == "activation_weighted_svd":
        if training_activation is None:
            raise ValueError("training activations are required for activation-weighted SVD")
        weights = np.asarray(training_activation, dtype=np.float64)
        coordinates = values.shape[0] // 2
        if weights.ndim == 2 and weights.shape[1] == coordinates:
            rms = np.sqrt(np.mean(weights * weights, axis=0))
            weights = np.concatenate((rms, rms))
        weights = weights.reshape(-1)
        if weights.size != values.shape[0]:
            raise ValueError("training activations do not match factor rows")
        _, _, vt = np.linalg.svd(values.astype(np.float64) * weights[:, None], full_matrices=False)
        return np.asarray(values @ vt[:r].T, np.float32)
    raise ValueError(f"unknown low-rank method: {method}")


@dataclass(frozen=True)
class EncodedFactor:
    values: np.ndarray
    scales: np.ndarray | None
    encoding: str

    @property
    def metadata_bytes(self) -> int:
        return int(self.values.nbytes + (0 if self.scales is None else self.scales.nbytes))

    def decode(self) -> np.ndarray:
        if self.encoding == "fp16":
            return self.values.astype(np.float32)
        if self.encoding == "int8_per_column":
            assert self.scales is not None
            return self.values.astype(np.float32) * self.scales.astype(np.float32)[None, :]
        raise ValueError(self.encoding)


def encode_factor(factor: np.ndarray, encoding: str) -> EncodedFactor:
    values = np.asarray(factor, dtype=np.float32)
    if encoding == "fp16":
        return EncodedFactor(values.astype(np.float16), None, "fp16")
    if encoding == "int8":
        scale = np.maximum(np.max(np.abs(values), axis=0) / 127.0, np.finfo(np.float16).tiny)
        quantized = np.clip(np.rint(values / scale[None, :]), -127, 127).astype(np.int8)
        return EncodedFactor(quantized, scale.astype(np.float16), "int8_per_column")
    raise ValueError(f"unknown factor encoding: {encoding}")


def low_rank_marginal_fixed_greedy(
    factor: np.ndarray | EncodedFactor,
    exact_diag: np.ndarray,
    activation: np.ndarray,
) -> list[BitAction]:
    u = factor.decode() if isinstance(factor, EncodedFactor) else np.asarray(factor, np.float32)
    if u.shape[0] % 2:
        raise ValueError("factor must contain two equal action planes")
    n = u.shape[0] // 2
    a = expanded_activation(activation, n)
    diag = np.asarray(exact_diag, np.float32).reshape(-1)
    if diag.size != 2 * n:
        raise ValueError("exact diagonal does not match factor")
    p = u.T @ a
    depth = np.zeros(n, np.uint8)
    selected = np.zeros(2 * n, bool)
    result: list[BitAction] = []
    for _ in range(2 * n):
        marginal = 2.0 * a * (u @ p) - a * a * diag
        marginal[~_eligible(depth) | selected] = -np.inf
        chosen = int(np.argmax(marginal))
        if not np.isfinite(marginal[chosen]):
            break
        result.append(_action(chosen, n, float(marginal[chosen])))
        selected[chosen] = True
        depth[chosen % n] = chosen // n + 1
        p -= float(a[chosen]) * u[chosen]
    return result


@dataclass(frozen=True)
class PageSelection:
    actions: tuple[BitAction, ...]
    pages: tuple[int, ...]
    page_gains: tuple[float, ...]


def separate_plane_pages(layout: np.ndarray, *, per_page: int) -> dict[int, tuple[int, ...]]:
    """Map page IDs to stage-major action IDs for a separate-plane layout."""
    order = np.asarray(layout, dtype=np.int64)
    n = len(order)
    if sorted(order.tolist()) != list(range(n)) or per_page < 1:
        raise ValueError("layout must be a permutation and per_page positive")
    pages: dict[int, tuple[int, ...]] = {}
    page_id = 0
    for stage in range(2):
        for start in range(0, n, per_page):
            pages[page_id] = tuple((stage * n + order[start:start + per_page]).tolist())
            page_id += 1
    return pages


def _best_page_mask(
    ids: Sequence[int], gram: np.ndarray, a: np.ndarray,
    correlation: np.ndarray, depth: np.ndarray,
) -> tuple[float, tuple[int, ...]]:
    n = len(depth)
    candidate = tuple(
        int(j) for j in ids
        if int(j) // n + 1 == int(depth[int(j) % n]) + 1
    )
    if not candidate:
        return 0.0, ()
    selected = np.asarray(candidate, dtype=np.int64)
    masks = _binary_masks(len(candidate))
    coefficients = masks * a[selected][None, :]
    local = gram[np.ix_(selected, selected)]
    gains = 2.0 * (coefficients @ correlation[selected])
    gains -= np.einsum("bi,ij,bj->b", coefficients, local, coefficients, optimize=True)
    position = int(np.argmax(gains))
    if gains[position] <= 0.0:
        return 0.0, ()
    subset = tuple(selected[masks[position].astype(bool)].tolist())
    return float(gains[position]), subset


@lru_cache(maxsize=8)
def _binary_masks(width: int) -> np.ndarray:
    """All nonempty page-application masks in deterministic integer order."""
    values = np.arange(1, 1 << width, dtype=np.uint16)[:, None]
    bits = np.arange(width, dtype=np.uint16)[None, :]
    result = ((values >> bits) & 1).astype(np.float64)
    result.setflags(write=False)
    return result


def page_aware_fixed_greedy(
    gram: np.ndarray, activation: np.ndarray,
    pages: Mapping[int, Sequence[int]], page_budget: int,
) -> PageSelection:
    """Select physical pages by their exact best feasible contained mask."""
    k = np.asarray(gram, dtype=np.float64)
    n = k.shape[0] // 2
    a = expanded_activation(activation, n).astype(np.float64)
    correlation = k @ a
    depth = np.zeros(n, np.uint8)
    paid: list[int] = []
    gains: list[float] = []
    actions: list[BitAction] = []
    remaining = set(int(p) for p in pages)

    def apply_mask(mask: Sequence[int]) -> float:
        recovered = 0.0
        for j in sorted(mask, key=lambda value: value // n):
            marginal = 2.0 * a[j] * correlation[j] - a[j] * a[j] * k[j, j]
            actions.append(_action(j, n, marginal))
            correlation[:] -= a[j] * k[:, j]
            depth[j % n] = j // n + 1
            recovered += float(marginal)
        return recovered

    def exhaust_paid_pages() -> float:
        """Apply newly feasible positive masks without charging another page."""
        recovered = 0.0
        while True:
            best_paid: tuple[float, int, tuple[int, ...]] | None = None
            for page in paid:
                gain, mask = _best_page_mask(pages[page], k, a, correlation, depth)
                candidate = (gain, -page, mask)
                if best_paid is None or candidate > best_paid:
                    best_paid = candidate
            if best_paid is None or best_paid[0] <= 0.0:
                return recovered
            recovered += apply_mask(best_paid[2])

    for _ in range(min(int(page_budget), len(remaining))):
        best: tuple[float, int, tuple[int, ...]] | None = None
        for page in sorted(remaining):
            gain, mask = _best_page_mask(pages[page], k, a, correlation, depth)
            candidate = (gain, -page, mask)
            if best is None or candidate > best:
                best = candidate
        if best is None or best[0] <= 0.0:
            break
        gain, negative_page, mask = best
        page = -negative_page
        remaining.remove(page)
        paid.append(page)
        paid_gain = apply_mask(mask)
        paid_gain += exhaust_paid_pages()
        gains.append(float(paid_gain))
    return PageSelection(tuple(actions), tuple(paid), tuple(gains))
