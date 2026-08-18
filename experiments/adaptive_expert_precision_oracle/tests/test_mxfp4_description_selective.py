from __future__ import annotations

import numpy as np

from oracle_study.mxfp4_description_selective import (
    action_page_ids,
    is_exact_mask,
    mask_descriptions,
    reconstruct_selected_correction,
    select_description_actions,
)


def test_b_can_be_selected_before_a() -> None:
    target = np.asarray([[4.0, 0.0], [0.0, 3.0]], np.float32)
    single = {
        "a": np.asarray([[0.1, 0.0], [0.0, 0.2]], np.float32),
        "b": np.asarray([[3.9, 0.0], [0.0, 2.9]], np.float32),
    }
    actions, pages, state = select_description_actions(
        target, single, np.arange(2), payload_bytes=64, page_size=64,
        byte_budget=64, available_descriptions=("a", "b"),
    )
    assert actions[0].add_descriptions == ("b",)
    assert mask_descriptions(int(state[actions[0].coordinate])) == ("b",)
    assert len(pages) == 1


def test_parity_can_be_best_one_bit_description() -> None:
    target = np.asarray([[5.0, 0.0]], np.float32)
    single = {
        "a": np.asarray([[1.0, 0.0]], np.float32),
        "b": np.asarray([[2.0, 0.0]], np.float32),
        "q": np.asarray([[4.8, 0.0]], np.float32),
    }
    actions, _, state = select_description_actions(
        target, single, np.arange(1), payload_bytes=64, page_size=64,
        byte_budget=64, available_descriptions=("a", "b", "q"),
    )
    assert actions[0].add_descriptions == ("q",)
    assert mask_descriptions(int(state[0])) == ("q",)


def test_any_two_selected_descriptions_decode_exact_contribution() -> None:
    target = np.asarray([[4.0, -2.0]], np.float32)
    single = {
        "a": np.asarray([[3.0, -1.0]], np.float32),
        "b": np.asarray([[1.0, -1.5]], np.float32),
        "q": np.asarray([[2.0, -1.8]], np.float32),
    }
    actions, pages, state = select_description_actions(
        target, single, np.arange(1), payload_bytes=64, page_size=64,
        byte_budget=128, available_descriptions=("a", "b", "q"),
    )
    assert is_exact_mask(int(state[0]))
    assert len(pages) == 2
    assert np.array_equal(reconstruct_selected_correction(target, single, state), target[0])
    charged = set(sum(action_page_ids(actions, np.arange(1), 64, 64, ("a", "b", "q")), []))
    assert charged == pages


def test_page_budget_charges_exact_selected_description_ids() -> None:
    coordinates = 16
    target = np.ones((coordinates, 3), np.float32)
    single = {
        "a": np.full_like(target, 0.9),
        "b": np.full_like(target, 0.8),
    }
    actions, pages, _ = select_description_actions(
        target, single, np.arange(coordinates), payload_bytes=64, page_size=512,
        byte_budget=512, available_descriptions=("a", "b"),
    )
    charged = set(sum(action_page_ids(actions, np.arange(coordinates), 64, 512, ("a", "b")), []))
    assert charged == pages
    assert len(pages) == 1
    assert all(action.add_descriptions == ("a",) for action in actions)
