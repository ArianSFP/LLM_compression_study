from __future__ import annotations

import itertools

import numpy as np
import pytest

from oracle_study.mxfp4_descriptions import (
    DESCRIPTIONS,
    MultipleDescriptionTree,
    description_bits,
    description_distortion,
)
from oracle_study.mxfp4_embed import NATURAL_ORDER, fit_tree


def hierarchy() -> MultipleDescriptionTree:
    masses = np.arange(1, 17, dtype=np.float64)
    return MultipleDescriptionTree.fit(fit_tree(NATURAL_ORDER, masses, "natural", "free_fp16"), masses)


def test_each_description_is_a_distinct_balanced_partition() -> None:
    value = hierarchy()
    parent, a, b, q = description_bits(value.base, np.arange(16, dtype=np.uint8))
    assert np.array_equal(q, a ^ b)
    partitions = set()
    for bits in (a, b, q):
        signature = []
        for p in range(4):
            assert np.array_equal(np.bincount(bits[parent == p], minlength=2), [2, 2])
            signature.append(tuple(bits[parent == p]))
        partitions.add(tuple(signature))
    assert len(partitions) == 3


@pytest.mark.parametrize("first,second", itertools.combinations(DESCRIPTIONS, 2))
def test_any_two_descriptions_reconstruct_exact_leaf(first: str, second: str) -> None:
    value = hierarchy()
    leaves = np.arange(16, dtype=np.uint8)
    parent, a, b, q = description_bits(value.base, leaves)
    bits = {"a": a, "b": b, "q": q}
    actual = value.reconstruct_leaf_codes(parent, first, bits[first], second, bits[second])
    assert np.array_equal(actual, leaves)
    assert np.array_equal(value.decode_normalized(leaves, (first, second)), value.decode_normalized(leaves, ("a", "b")))


def test_single_description_tables_are_independently_fitted() -> None:
    value = hierarchy()
    assert not np.array_equal(value.c3a, value.c3b)
    assert not np.array_equal(value.c3a, value.c3q)
    for name in DESCRIPTIONS:
        assert np.isfinite(description_distortion(value, np.ones(16), name))


def test_invalid_description_sets_fail_closed() -> None:
    value = hierarchy()
    leaves = np.arange(16, dtype=np.uint8)
    with pytest.raises(ValueError):
        value.decode_normalized(leaves, ("bad",))
    with pytest.raises(ValueError):
        value.decode_normalized(leaves, ("a", "b", "q"))
    with pytest.raises(ValueError):
        value.reconstruct_leaf_codes(np.zeros(1, np.uint8), "a", np.zeros(1, np.uint8), "a", np.zeros(1, np.uint8))
