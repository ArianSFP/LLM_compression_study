from __future__ import annotations

from types import SimpleNamespace

import torch

from oracle_study.d1_decode import (
    cache_state_metrics,
    cache_tensor_items,
    clone_decode_cache,
    mutation_safe_cached_autograd,
    tensor_state_metrics,
)


def _fake_cache() -> SimpleNamespace:
    linear = SimpleNamespace(
        keys=torch.tensor([]),
        values=torch.tensor([]),
        conv_states={0: torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)},
        recurrent_states={0: torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)},
    )
    attention = SimpleNamespace(
        keys=torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2),
        values=torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2) + 1,
    )
    return SimpleNamespace(layers=[linear, attention])


def test_cache_clone_is_equal_and_storage_isolated() -> None:
    cache = _fake_cache()
    cloned = clone_decode_cache(cache)
    original_items = dict(cache_tensor_items(cache))
    cloned_items = dict(cache_tensor_items(cloned))
    assert original_items.keys() == cloned_items.keys()
    for name in original_items:
        assert torch.equal(original_items[name], cloned_items[name])
        assert original_items[name].data_ptr() != cloned_items[name].data_ptr()
    cloned.layers[0].recurrent_states[0].add_(1)
    assert not torch.equal(
        cache.layers[0].recurrent_states[0],
        cloned.layers[0].recurrent_states[0],
    )


def test_cache_state_metrics_can_scope_one_layer() -> None:
    reference = _fake_cache()
    candidate = clone_decode_cache(reference)
    exact = cache_state_metrics(reference, candidate)
    assert exact.tensors == 4
    assert exact.bit_identical
    assert exact.max_abs == 0.0
    candidate.layers[1].values[..., -1, :].add_(2)
    attention = cache_state_metrics(reference, candidate, layer=1)
    assert attention.tensors == 2
    assert not attention.bit_identical
    assert attention.max_abs == 2.0
    assert attention.mse > 0.0
    assert cache_state_metrics(reference, candidate, layer=0).bit_identical


def test_tensor_state_metrics_reports_exact_and_mse() -> None:
    reference = torch.zeros(2, 3)
    candidate = reference.clone()
    candidate[0, 0] = 3
    metrics = tensor_state_metrics(reference, candidate, "hidden")
    assert not metrics["hidden_bit_identical"]
    assert metrics["hidden_max_abs"] == 3.0
    assert metrics["hidden_mse"] == 1.5


def test_mutation_safe_cached_autograd_preserves_precommit_value() -> None:
    current = torch.tensor([2.0], requires_grad=True)
    cache_state = torch.tensor([3.0])
    with mutation_safe_cached_autograd():
        result = current * cache_state
        cache_state.copy_(torch.tensor([11.0]))
        gradient = torch.autograd.grad(result, current)[0]
    assert torch.equal(gradient, torch.tensor([3.0]))
