"""Cache-safe primitives for exact-prefill, single-token D1 decode studies."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class CacheStateMetrics:
    tensors: int
    coordinates: int
    bit_identical: bool
    max_abs: float
    mse: float

    def to_dict(self, prefix: str = "cache") -> dict[str, int | bool | float]:
        return {
            f"{prefix}_tensors": int(self.tensors),
            f"{prefix}_coordinates": int(self.coordinates),
            f"{prefix}_bit_identical": bool(self.bit_identical),
            f"{prefix}_max_abs": float(self.max_abs),
            f"{prefix}_mse": float(self.mse),
        }


def mutation_safe_cached_autograd() -> Any:
    """Protect a native cached forward's saved tensors from in-place commits.

    Qwen's one-token DeltaNet kernel updates its convolution and recurrent
    cache tensors during the forward.  Reverse-mode autograd also needs the
    pre-commit values.  Saved-tensor hooks retain private copies for backward
    without replacing or approximating the native forward.
    """

    import torch

    return torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: tensor.detach().clone(),
        lambda tensor: tensor,
    )


def clone_decode_cache(cache: Any) -> Any:
    """Deep-copy a Transformers hybrid cache, including every tensor storage."""

    result = copy.deepcopy(cache)
    source = dict(cache_tensor_items(cache))
    cloned = dict(cache_tensor_items(result))
    if source.keys() != cloned.keys():
        raise RuntimeError("decode cache clone changed its tensor inventory")
    for name in source:
        left, right = source[name], cloned[name]
        if not left.equal(right):
            raise RuntimeError(f"decode cache clone changed tensor {name}")
        if left.numel() and left.data_ptr() == right.data_ptr():
            raise RuntimeError(f"decode cache clone aliases tensor {name}")
    return result


def _initialized(value: Any) -> bool:
    return value is not None and hasattr(value, "numel") and int(value.numel()) > 0


def cache_tensor_items(cache: Any, layer: int | None = None) -> Iterator[tuple[str, Any]]:
    """Yield initialized K/V, convolution, and recurrent tensors deterministically."""

    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("decode cache does not expose layers")
    selected = range(len(layers)) if layer is None else (int(layer),)
    for layer_index in selected:
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(f"decode cache layer is absent: {layer_index}")
        item = layers[layer_index]
        for field in ("keys", "values"):
            value = getattr(item, field, None)
            if _initialized(value):
                yield f"layer_{layer_index:02d}.{field}", value
        for field in ("conv_states", "recurrent_states"):
            states = getattr(item, field, None)
            if states is None:
                continue
            iterator = states.items() if hasattr(states, "items") else enumerate(states)
            for state_index, value in sorted(iterator, key=lambda pair: int(pair[0])):
                if _initialized(value):
                    yield f"layer_{layer_index:02d}.{field}.{int(state_index)}", value


def cache_state_metrics(reference: Any, candidate: Any, layer: int | None = None) -> CacheStateMetrics:
    """Compare cache state without flattening the potentially large tensors."""

    reference_items = dict(cache_tensor_items(reference, layer))
    candidate_items = dict(cache_tensor_items(candidate, layer))
    if reference_items.keys() != candidate_items.keys():
        raise RuntimeError("decode cache tensor inventories differ")
    coordinates = 0
    squared_error = 0.0
    maximum = 0.0
    identical = True
    for name in reference_items:
        left = reference_items[name]
        right = candidate_items[name]
        if tuple(left.shape) != tuple(right.shape):
            raise RuntimeError(f"decode cache tensor shape differs: {name}")
        identical = identical and bool(left.equal(right))
        difference = left.detach().float() - right.detach().float()
        count = int(difference.numel())
        coordinates += count
        if count:
            maximum = max(maximum, float(difference.abs().max().item()))
            squared_error += float((difference.double() ** 2).sum().item())
    return CacheStateMetrics(
        tensors=len(reference_items),
        coordinates=coordinates,
        bit_identical=identical,
        max_abs=maximum,
        mse=(squared_error / coordinates if coordinates else 0.0),
    )


def tensor_state_metrics(reference: Any, candidate: Any, prefix: str) -> dict[str, float | bool]:
    """Return exact and floating error measures for equal-shaped tensors."""

    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(f"{prefix} tensor shape differs")
    difference = reference.detach().float() - candidate.detach().float()
    return {
        f"{prefix}_bit_identical": bool(reference.equal(candidate)),
        f"{prefix}_max_abs": float(difference.abs().max().item()) if difference.numel() else 0.0,
        f"{prefix}_mse": float((difference.double() ** 2).mean().item()) if difference.numel() else 0.0,
    }
