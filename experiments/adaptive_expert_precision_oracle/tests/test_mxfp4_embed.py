from __future__ import annotations

import numpy as np
import pytest

from oracle_study.mxfp4_embed import (
    EmbeddedTree,
    MXFP4LeafTensor,
    NATURAL_ORDER,
    decode_e8m0,
    fit_tree,
    leaf_importance_statistics,
    maps_from_order,
    pack_1bit,
    pack_2bit,
    pack_leaf_codes,
    select_tree,
    tree_orders,
    unpack_1bit,
    unpack_2bit,
    unpack_leaf_codes,
)


def sample_tensor() -> MXFP4LeafTensor:
    codes = np.tile(np.arange(16, dtype=np.uint8), (2, 4))
    scales = np.asarray([[127, 128], [126, 129]], dtype=np.uint8)
    return MXFP4LeafTensor(codes, scales, "synthetic")


def test_exact_packed_leaf_roundtrip() -> None:
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 16, size=(7, 64), dtype=np.uint8)
    packed = pack_leaf_codes(codes)
    assert np.array_equal(unpack_leaf_codes(packed, codes.shape), codes)


def test_bitplane_roundtrips() -> None:
    rng = np.random.default_rng(8)
    bits = rng.integers(0, 2, size=(5, 64), dtype=np.uint8)
    parents = rng.integers(0, 4, size=(5, 64), dtype=np.uint8)
    assert np.array_equal(unpack_1bit(pack_1bit(bits), 64), bits)
    assert np.array_equal(unpack_2bit(pack_2bit(parents), 64), parents)


def test_e8m0_scale_decode() -> None:
    actual = decode_e8m0(np.asarray([126, 127, 128], dtype=np.uint8))
    assert np.array_equal(actual, np.asarray([0.5, 1.0, 2.0], dtype=np.float32))


def test_full_hierarchy_is_exact() -> None:
    tensor = sample_tensor()
    masses = leaf_importance_statistics(tensor)
    tree = fit_tree(NATURAL_ORDER, masses, "natural", "free_fp16")
    assert np.array_equal(tree.reconstruct_leaf_codes(tensor.codes), tensor.codes)
    assert np.array_equal(tree.decode(tensor, 4), tensor.dequantize())
    assert np.array_equal(pack_leaf_codes(tree.reconstruct_leaf_codes(tensor.codes)), tensor.exact_streams()[0])


def test_tree_is_balanced_bijection() -> None:
    parent, r1, r2 = maps_from_order(NATURAL_ORDER)
    triples = {(int(parent[i]), int(r1[i]), int(r2[i])) for i in range(16)}
    assert len(triples) == 16
    assert np.array_equal(np.bincount(parent, minlength=4), np.full(4, 4))
    with pytest.raises(ValueError):
        EmbeddedTree(parent, r1, np.zeros(16, dtype=np.uint8), np.zeros(4), np.zeros((4, 2)), "bad", "e2m1")


def test_centroid_selection_is_deterministic_and_hardware_constrained() -> None:
    masses = np.arange(1, 17, dtype=np.float64)
    orders = tree_orders("unrestricted", seed=91, random_candidates=32)
    first = select_tree(orders, masses, "learned", "e2m1")
    second = select_tree(orders, masses, "learned", "e2m1")
    assert first.as_dict() == second.as_dict()
    allowed = {-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    assert set(map(float, first.c2)).issubset(allowed)
    assert set(map(float, first.c3.reshape(-1))).issubset(allowed)


def test_resident_and_suffix_byte_rates() -> None:
    tensor = sample_tensor()
    facts = tensor.storage_bytes()
    assert facts["effective_bpw"] == pytest.approx(4.25)
    weights = facts["weights"]
    resident = weights // 4 + weights // 32
    suffix = weights // 8
    assert 8 * resident / weights == pytest.approx(2.25)
    assert 8 * suffix / weights == pytest.approx(1.0)
