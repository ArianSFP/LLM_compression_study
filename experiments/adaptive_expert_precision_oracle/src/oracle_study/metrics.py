"""Per-invocation quality metrics and exact marginal support selection."""

from __future__ import annotations

import numpy as np
import torch


def quadratic_energy(error: np.ndarray, proxy: np.ndarray | None = None, beta: float = 0.0) -> np.ndarray:
    value = np.sum(np.asarray(error, dtype=np.float64) ** 2, axis=-1)
    if proxy is not None and beta:
        projected = np.asarray(error, dtype=np.float64) @ np.asarray(proxy, dtype=np.float64)
        value = value + beta * np.sum(projected**2, axis=-1)
    return value


def metric_rows(
    reference: np.ndarray,
    base: np.ndarray,
    approximation: np.ndarray,
    proxy: np.ndarray | None = None,
    beta: float = 0.0,
) -> dict[str, np.ndarray]:
    reference = np.atleast_2d(np.asarray(reference, dtype=np.float32))
    base = np.atleast_2d(np.asarray(base, dtype=np.float32))
    approximation = np.atleast_2d(np.asarray(approximation, dtype=np.float32))
    error = reference - approximation
    denominator = quadratic_energy(reference - base, proxy, beta)
    numerator = quadratic_energy(error, proxy, beta)
    recovery = 1.0 - numerator / np.maximum(denominator, 1e-20)
    ref_norm = np.linalg.norm(reference, axis=1)
    err_norm = np.linalg.norm(error, axis=1)
    cosine = np.sum(reference * approximation, axis=1) / np.maximum(
        ref_norm * np.linalg.norm(approximation, axis=1), 1e-20
    )
    return {
        "recovery": recovery,
        "relative_output_error": err_norm / np.maximum(ref_norm, 1e-20),
        "cosine_similarity": cosine,
        "absolute_error": np.sqrt(np.mean(error.astype(np.float64) ** 2, axis=1)),
        "damage": numerator,
        "base_damage": denominator,
    }


def ranked_approximations(
    atoms: torch.Tensor,
    codes: torch.Tensor,
    reference_correction: torch.Tensor,
    counts: list[int],
    score_rule: str = "energy",
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Return activation-specific ranked partial atom sums for a batch."""
    norms = torch.sum(atoms * atoms, dim=0)
    if score_rule == "abs":
        scores = torch.abs(codes)
    elif score_rule == "energy":
        scores = codes * codes * norms[None, :]
    else:
        raise ValueError(score_rule)
    maximum = min(max(counts, default=0), atoms.shape[1])
    if maximum == 0:
        zeros = torch.zeros_like(reference_correction)
        return {count: zeros for count in counts}, torch.empty((len(codes), 0), dtype=torch.long, device=codes.device)
    indices = torch.topk(scores, maximum, dim=1, largest=True, sorted=True).indices
    selected_codes = torch.gather(codes, 1, indices)
    selected_atoms = atoms.T[indices]
    cumulative = torch.cumsum(selected_atoms * selected_codes[:, :, None], dim=1)
    result: dict[int, torch.Tensor] = {}
    zeros = torch.zeros_like(reference_correction)
    for count in counts:
        effective = min(count, maximum)
        result[count] = zeros if effective == 0 else cumulative[:, effective - 1]
    return result, indices


def exact_greedy_approximations(
    atoms: torch.Tensor,
    codes: torch.Tensor,
    correction: torch.Tensor,
    counts: list[int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Exact greedy marginal reduction under local squared error."""
    n, width = codes.shape
    maximum = min(max(counts, default=0), width)
    residual = correction.clone()
    approximation = torch.zeros_like(correction)
    used = torch.zeros((n, width), dtype=torch.bool, device=codes.device)
    norm = torch.sum(atoms * atoms, dim=0)[None, :]
    choices = torch.empty((n, maximum), dtype=torch.long, device=codes.device)
    snapshots: dict[int, torch.Tensor] = {0: approximation.clone()}
    wanted = set(min(v, maximum) for v in counts)
    rows = torch.arange(n, device=codes.device)
    for step in range(1, maximum + 1):
        dot = residual @ atoms
        gain = 2.0 * codes * dot - codes * codes * norm
        gain = gain.masked_fill(used, -torch.inf)
        selected = torch.argmax(gain, dim=1)
        choices[:, step - 1] = selected
        used[rows, selected] = True
        contribution = atoms[:, selected].T * codes[rows, selected][:, None]
        approximation = approximation + contribution
        residual = residual - contribution
        if step in wanted:
            snapshots[step] = approximation.clone()
    return {count: snapshots[min(count, maximum)] for count in counts}, choices


def router_metrics(reference_logits: np.ndarray, approximate_logits: np.ndarray, k: int = 8) -> dict[str, np.ndarray]:
    ref = np.asarray(reference_logits, dtype=np.float64)
    approx = np.asarray(approximate_logits, dtype=np.float64)
    ref_ids = np.argpartition(ref, -k, axis=1)[:, -k:]
    app_ids = np.argpartition(approx, -k, axis=1)[:, -k:]
    recall, jaccard = [], []
    for left, right in zip(ref_ids, app_ids):
        intersection = len(set(left.tolist()) & set(right.tolist()))
        recall.append(intersection / k)
        jaccard.append(intersection / (2 * k - intersection))
    ref_sorted = np.sort(ref, axis=1)
    app_sorted = np.sort(approx, axis=1)
    return {
        "top8_recall": np.asarray(recall),
        "top8_jaccard": np.asarray(jaccard),
        "boundary_margin_reference": ref_sorted[:, -k] - ref_sorted[:, -k - 1],
        "boundary_margin_approximate": app_sorted[:, -k] - app_sorted[:, -k - 1],
        "boundary_flip": np.asarray([set(a.tolist()) != set(b.tolist()) for a, b in zip(ref_ids, app_ids)], dtype=np.float64),
    }
