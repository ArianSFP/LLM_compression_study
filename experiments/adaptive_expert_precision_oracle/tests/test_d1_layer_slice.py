from __future__ import annotations

import numpy as np
import pytest

from oracle_study.d1_layer_slice import (
    D1SliceParity,
    _D1LayerSlice,
    slice_parity,
    topk_membership,
)


def test_topk_membership_is_stable_for_ties() -> None:
    logits = np.asarray([[1.0, 3.0, 3.0, 0.0], [0.0, -1.0, 2.0, 1.0]])
    np.testing.assert_array_equal(
        topk_membership(logits, 2),
        np.asarray([[1, 2], [2, 3]]),
    )


def test_slice_parity_distinguishes_order_from_set_membership() -> None:
    hidden = np.asarray([[1.0, 2.0]], np.float32)
    reference = np.asarray([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    candidate = reference.copy()
    candidate[0, 0], candidate[0, 1] = candidate[0, 1], candidate[0, 0]
    parity = slice_parity(hidden, hidden, candidate, reference)
    assert parity.current_hidden_bit_identical
    assert not parity.next_router_bit_identical
    assert parity.next_top8_exact_fraction == 0.0
    assert parity.next_top8_set_exact_fraction == 1.0


def test_slice_parity_rejects_shape_drift() -> None:
    with pytest.raises(ValueError, match="shapes changed"):
        slice_parity(
            np.zeros((2, 2)), np.zeros((1, 2)),
            np.zeros((2, 9)), np.zeros((2, 9)),
        )


def test_d1_slice_parity_validates_fractions() -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        D1SliceParity(True, 0.0, True, 0.0, 1.01, 1.0)
def test_decode_prefix_cache_advances_one_token_at_a_time(monkeypatch) -> None:
    import torch
    import transformers

    class FakeCache:
        def __init__(self, config) -> None:
            self.config = config

    class RecordingMixer:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, *, hidden_states, cache_params, attention_mask):
            assert attention_mask is None
            self.calls.append((hidden_states.detach().clone(), cache_params))

    monkeypatch.setattr(transformers, "DynamicCache", FakeCache)
    mixer = RecordingMixer()
    layer_slice = _D1LayerSlice(
        layer=6,
        current_shared=None,
        current_shared_gate=None,
        next_input_norm=lambda value: value + 1,
        next_mixer=mixer,
        next_mixer_type="linear_attention",
        next_config=object(),
        next_rotary=None,
        next_post_norm=None,
        next_router=None,
        device="cpu",
    )
    hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    cache = layer_slice.decode_prefix_cache(hidden)
    assert isinstance(cache, FakeCache)
    assert len(mixer.calls) == 3
    for position, (observed, observed_cache) in enumerate(mixer.calls):
        assert observed_cache is cache
        assert tuple(observed.shape) == (1, 1, 4)
        torch.testing.assert_close(
            observed,
            hidden[position : position + 1].unsqueeze(0) + 1,
        )
